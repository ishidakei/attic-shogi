//! Time management — a faithful port of the reference `TimeManagement`
//! (`upstream YaneuraOu @ 76d58ef`, `source/timeman.{h,cpp}` compiled under
//! `USE_TIME_MANAGEMENT`, non-DEEP build).
//!
//! The reference class computes, for one `go`, an **optimum**, **maximum**, and
//! **minimum** think time in milliseconds from the clock (`time[us]`), the
//! Fischer increment (`inc[us]`), the byoyomi (`byoyomi[us]`), the game ply, and
//! the engine's time options. The search then dynamically shrinks the deadline
//! toward `optimum` based on eval stability (see [`crate::qsearch`]'s iterative
//! deepening) and stops at [`TimeManagement::search_end`], which is filled by
//! [`TimeManagement::set_search_end`] and enforced in `check_time`.
//!
//! All internal arithmetic is in `i64` milliseconds, mirroring the reference
//! `TimePoint`; the same `(int)` truncation / `float` (`f32`) / `double` (`f64`)
//! points as the C++ code are preserved so the computed budgets match bit-for-bit
//! given the same inputs. Wall-clock elapsed is measured from an [`Instant`]
//! captured when the `go` arrived (the reference `limits.startTime`).
//!
//! Ponder is wired in the search / USI layers: on a `ponderhit`
//! the search stamps [`TimeManagement::ponderhit_time`] to `now`, so the
//! `startTime - ponderhitTime` terms in [`TimeManagement::set_search_end`] carry
//! real (non-zero) values and the used time is counted from the ponderhit. With no
//! ponder `ponderhitTime == startTime` and those terms vanish, exercising the same
//! paths in their ponder-off shape.

use std::time::Instant;

use crate::book::Prng;

/// The reference `MoveHorizon` (`timeman.cpp`): the assumed number of plies
/// still to play when planning the time budget.
const MOVE_HORIZON: i32 = 160;

/// The raw inputs [`TimeManagement::init`] needs, extracted by the USI driver
/// from the `go` limits, the engine options, and the root position. Keeping the
/// input primitive (rather than the protocol `GoLimits`) preserves the layering
/// rule that Search never depends on Protocol.
#[derive(Clone, Copy, Debug)]
pub struct TimeInput {
    /// `limits.time[us]` — the side-to-move's remaining main clock [ms].
    pub time_us: i64,
    /// `limits.inc[us]` — the side-to-move's Fischer increment [ms].
    pub inc_us: i64,
    /// `limits.byoyomi[us]` — the side-to-move's byoyomi [ms].
    pub byoyomi_us: i64,
    /// `limits.movetime` [ms] (`0` when not a `go movetime`).
    pub movetime: i64,
    /// `limits.rtime` [ms] (`0` when not a `go rtime`).
    pub rtime: i64,
    /// `options["NetworkDelay"]` [ms].
    pub network_delay: i64,
    /// `options["NetworkDelay2"]` [ms].
    pub network_delay2: i64,
    /// `options["MinimumThinkingTime"]` [ms].
    pub minimum_thinking_time: i64,
    /// `options["SlowMover"]` — percentage multiplier on the optimum time.
    pub slow_mover: i64,
    /// `options["RoundUpToFullSecond"]`.
    pub round_up_to_fullsecond: bool,
    /// `options["USI_Ponder"]`.
    pub usi_ponder: bool,
    /// `options["Stochastic_Ponder"]`.
    pub stochastic_ponder: bool,
    /// `ply` — the root's game ply (`rootPos.game_ply()`; 1 at the hirate start).
    pub ply: i32,
    /// `max_moves_to_draw` — the game ply past which a draw is adjudicated
    /// (already the `0 → 100000` unlimited remap).
    pub max_moves_to_draw: i32,
    /// The instant the `go` arrived (`limits.startTime`), the origin for
    /// [`TimeManagement::elapsed`].
    pub start_time: Instant,
}

/// The reference `TimeManagement` state for one `go`.
#[derive(Clone, Debug)]
pub struct TimeManagement {
    /// `startTime` — the origin for [`Self::elapsed`] (`now() - startTime`).
    pub start_time: Instant,
    /// `ponderhitTime` — equal to `start_time` until a `ponderhit`, at which point
    /// the search stamps it to the ponderhit instant (`set_ponderhit`,
    /// `yaneuraou-search.cpp`). Used by [`Self::set_search_end`].
    pub ponderhit_time: Instant,
    /// `search_end` [ms from `start_time`]: `0` means "not yet decided"; once set,
    /// the search stops when `search_end <= elapsed` (`timeman.h`).
    pub search_end: i64,
    /// `isFinalPush` — in byoyomi with (almost) no main clock, spend it all
    /// (`timeman.cpp`); consumed by [`Self::set_search_end`].
    pub is_final_push: bool,
    /// True only for the `MTG <= 0` error path (`timeman.cpp`), so the
    /// driver can emit the reference `info string Error!` diagnostic.
    pub mtg_error: bool,

    minimum_time: i64,
    optimum_time: i64,
    maximum_time: i64,

    minimum_thinking_time: i64,
    network_delay: i64,
    remain_time: i64,
    round_up_to_fullsecond: bool,
}

impl TimeManagement {
    /// Compute the think-time budget for one `go` (`timeman.cpp`).
    pub fn init(input: &TimeInput, prng: &mut Prng) -> TimeManagement {
        let &TimeInput {
            time_us,
            inc_us,
            byoyomi_us,
            movetime,
            rtime,
            network_delay,
            network_delay2,
            minimum_thinking_time,
            slow_mover,
            round_up_to_fullsecond,
            usi_ponder,
            stochastic_ponder,
            ply,
            max_moves_to_draw,
            start_time,
        } = input;

        let mut tm = TimeManagement {
            start_time,
            ponderhit_time: start_time,
            search_end: 0,
            is_final_push: false,
            mtg_error: false,
            minimum_time: 0,
            optimum_time: 0,
            maximum_time: 0,
            minimum_thinking_time,
            network_delay,
            remain_time: 0,
            round_up_to_fullsecond,
        };

        // Remaining time this move must respect, minus the worst-case network
        // delay; floored so a spent clock cannot self-destruct
        // (`timeman.cpp`). Byoyomi is folded in here because it is
        // available for *this* move; the Fischer increment is deliberately NOT
        // (`/* + limits.inc[us] */` at the pin) — it is credited only after the
        // move has been played, so spending it now would overdraw the clock.
        let mut remain_time = time_us + byoyomi_us - network_delay2;
        remain_time = remain_time.max(if round_up_to_fullsecond { 100 } else { 1 });
        tm.remain_time = remain_time;

        // `go rtime`: a randomised minimum-think budget, decaying with ply, used
        // for self-play variety (`timeman.cpp`).
        if rtime != 0 {
            let mut r = rtime;
            if ply != 0 {
                let bound = (r as f32 * 0.5).min(r as f32 * 10.0 / ply as f32) as i64;
                r += prng.rand(bound.max(0) as u64) as i64;
            }
            tm.remain_time = r;
            tm.minimum_time = r;
            tm.optimum_time = r;
            tm.maximum_time = r;
            return tm;
        }

        // `go movetime`: spend exactly the given time (`timeman.cpp`).
        if movetime != 0 {
            tm.remain_time = movetime;
            tm.minimum_time = movetime;
            tm.optimum_time = movetime;
            tm.maximum_time = movetime;
            return tm;
        }

        // Time-forfeit (sudden death): neither increment nor byoyomi
        // (`timeman.cpp`).
        let time_forfeit = inc_us == 0 && byoyomi_us == 0;

        // The planning horizon, wider early and narrower once out of the opening
        // (`timeman.cpp`).
        let move_horizon = if time_forfeit {
            MOVE_HORIZON + 40 - ply.min(40)
        } else {
            MOVE_HORIZON + 20 - ply.min(80)
        };

        // Own remaining moves until the draw horizon (`timeman.cpp`).
        let mtg = (max_moves_to_draw - ply + 2).min(move_horizon) / 2;

        if mtg <= 0 {
            // Should be unreachable given a sane MaxMovesToDraw; guard anyway
            // (`timeman.cpp`).
            tm.mtg_error = true;
            tm.minimum_time = 500;
            tm.optimum_time = 500;
            tm.maximum_time = 500;
            return tm;
        }
        if mtg == 1 {
            // Last move before the horizon: spend everything (`timeman.cpp`).
            tm.minimum_time = remain_time;
            tm.optimum_time = remain_time;
            tm.maximum_time = remain_time;
            return tm;
        }

        // Minimum think time floor (`timeman.cpp`).
        let minimum_time = (minimum_thinking_time - network_delay).max(if round_up_to_fullsecond {
            1000
        } else {
            1
        });
        tm.minimum_time = minimum_time;

        // Time estimated still available across the remaining moves
        // (`timeman.cpp`).
        let mut remain_estimate = time_us + inc_us * mtg as i64 + byoyomi_us * mtg as i64;
        if round_up_to_fullsecond {
            remain_estimate -= (mtg as i64 + 1) * 1000;
        }
        remain_estimate = remain_estimate.max(0);

        // optimum candidate (`timeman.cpp`).
        let t1 = minimum_time + remain_estimate / mtg as i64;

        // maximum candidate: up to `max_ratio`× the optimum, capped at 30% of the
        // remaining estimate (`timeman.cpp`).
        let mut max_ratio = 5.0f32;
        if time_forfeit {
            max_ratio = max_ratio.min((time_us as f32 / (60.0 * 1000.0)).max(1.0));
        }
        let mut t2 = minimum_time + (remain_estimate as f32 * max_ratio / mtg as f32) as i64;
        t2 = t2.min((remain_estimate as f64 * 0.3) as i64);

        // Fold in SlowMover and clamp to the remaining time (`timeman.cpp`).
        tm.optimum_time = t1.min(remain_time) * slow_mover / 100;
        tm.maximum_time = t2.min(remain_time);

        // Ponder bonus (`timeman.cpp`).
        if usi_ponder && !stochastic_ponder {
            tm.optimum_time += tm.optimum_time / 4;
        }

        // Byoyomi with (almost) no main clock: spend it all this move
        // (`timeman.cpp`).
        tm.is_final_push = false;
        if byoyomi_us != 0 && time_us < (byoyomi_us as f64 * 1.2) as i64 {
            let v = byoyomi_us + time_us;
            tm.minimum_time = v;
            tm.optimum_time = v;
            tm.maximum_time = v;
            tm.is_final_push = true;
        }

        // Final clamps: round up minimum/maximum and never exceed remain_time
        // (`timeman.cpp`).
        tm.minimum_time = tm.round_up(tm.minimum_time).min(remain_time);
        tm.optimum_time = tm.optimum_time.min(remain_time);
        tm.maximum_time = tm.round_up(tm.maximum_time).min(remain_time);

        tm
    }

    /// Round `t0` up to a whole second (subtracting the network delay), floored at
    /// `MinimumThinkingTime` and capped at `remain_time` (`timeman.cpp`).
    /// A no-op rounding when `RoundUpToFullSecond` is off.
    pub fn round_up(&self, t0: i64) -> i64 {
        if self.round_up_to_fullsecond {
            let mut t = (((t0 + 999) / 1000) * 1000).max(self.minimum_thinking_time);
            t -= self.network_delay;
            if t < t0 {
                t += 1000;
            }
            t.min(self.remain_time)
        } else {
            let mut t = t0.max(self.minimum_thinking_time);
            t -= self.network_delay;
            t.min(self.remain_time)
        }
    }

    /// Fix the search end time from the elapsed time `e` [ms] at which the search
    /// decided to stop (`timeman.cpp`). Rounds the used time up to a full
    /// second (honouring `isFinalPush` and the minimum think time) and stores the
    /// result as a `search_end` offset from `startTime`. Without ponder,
    /// `startTime == ponderhitTime`, so this reduces to
    /// `search_end = round_up(max(e, minimum()))`.
    pub fn set_search_end(&mut self, e: i64) {
        // `startTime - ponderhitTime` in ms (0 without ponder; <= 0 with).
        let start_minus_ponderhit = -(self
            .ponderhit_time
            .saturating_duration_since(self.start_time)
            .as_millis() as i64);
        let t1 = e + start_minus_ponderhit;
        let t2 = if self.is_final_push {
            self.minimum_time
        } else {
            self.minimum_time + start_minus_ponderhit
        };
        // `round_up(max(t1, t2)) + ponderhitTime - startTime`.
        self.search_end = self.round_up(t1.max(t2)) - start_minus_ponderhit;
    }

    /// Elapsed time [ms] since `startTime`, measured against `now`.
    pub fn elapsed_from(&self, now: Instant) -> i64 {
        now.saturating_duration_since(self.start_time).as_millis() as i64
    }

    /// Elapsed time [ms] since `startTime`, measured now.
    pub fn elapsed(&self) -> i64 {
        self.elapsed_from(Instant::now())
    }

    /// `optimum()` — the target think time [ms].
    pub fn optimum(&self) -> i64 {
        self.optimum_time
    }
    /// `maximum()` — the hard think-time ceiling [ms].
    pub fn maximum(&self) -> i64 {
        self.maximum_time
    }
    /// `minimum()` — the minimum think time [ms].
    pub fn minimum(&self) -> i64 {
        self.minimum_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `TimeInput` with the reference option defaults and no clock; individual
    /// tests override the fields they exercise.
    fn base() -> TimeInput {
        TimeInput {
            time_us: 0,
            inc_us: 0,
            byoyomi_us: 0,
            movetime: 0,
            rtime: 0,
            network_delay: 120,
            network_delay2: 1120,
            minimum_thinking_time: 2000,
            slow_mover: 100,
            round_up_to_fullsecond: true,
            usi_ponder: false,
            stochastic_ponder: false,
            ply: 1,
            max_moves_to_draw: 100_000,
            start_time: Instant::now(),
        }
    }

    fn init(input: &TimeInput) -> TimeManagement {
        let mut prng = Prng::new(1);
        TimeManagement::init(input, &mut prng)
    }

    #[test]
    fn movetime_sets_all_three_to_movetime() {
        let tm = init(&TimeInput {
            movetime: 3000,
            ..base()
        });
        assert_eq!(tm.minimum(), 3000);
        assert_eq!(tm.optimum(), 3000);
        assert_eq!(tm.maximum(), 3000);
    }

    #[test]
    fn byoyomi_10min_plus_10s() {
        // 10 min main + 10 s byoyomi, no increment. Hand-computed against
        // timeman.cpp with the default options and ply 1.
        let input = TimeInput {
            time_us: 600_000,
            byoyomi_us: 10_000,
            ply: 1,
            ..base()
        };
        let tm = init(&input);

        // remain_time = 600000 + 10000 - 1120 = 608880 (>= 100).
        // time_forfeit = false (byoyomi != 0).
        // move_horizon = 160 + 20 - min(1,80) = 179.
        // MTG = min(100000 - 1 + 2, 179) / 2 = 179/2 = 89.
        // minimum_time = max(2000 - 120, 1000) = 1880.
        // remain_estimate = 600000 + 0 + 10000*89 = 1490000
        //                 - (89 + 1)*1000 = 1400000 (>= 0).
        // t1 = 1880 + 1400000/89 = 1880 + 15730 = 17610.
        // max_ratio = 5.0 (not forfeit).
        // t2 = 1880 + (int)(1400000 * 5 / 89) = 1880 + 78651 = 80531.
        //    capped at (int)(1400000 * 0.3) = 420000 -> 80531.
        // optimum = min(17610, 608880) * 100/100 = 17610.
        // maximum = min(80531, 608880) = 80531.
        // not final push (600000 >= 10000*1.2 = 12000).
        // minimum = min(round_up(1880), 608880):
        //   round_up(1880): ((1880+999)/1000)*1000 = 2000; max(2000,2000)=2000;
        //   2000-120=1880; 1880 < 1880? no; min(1880, 608880)=1880.
        // optimum = min(17610, 608880) = 17610.
        // maximum = min(round_up(80531), 608880):
        //   round_up(80531): ((80531+999)/1000)*1000 = 81000; max(81000,2000)=81000;
        //   81000-120=80880; 80880 < 80531? no; 80880.
        assert_eq!(tm.minimum(), 1880);
        assert_eq!(tm.optimum(), 17610);
        assert_eq!(tm.maximum(), 80880);
        assert!(!tm.is_final_push);
    }

    #[test]
    fn fischer_time_plus_increment() {
        // 5 min + 5 s increment, ply 20.
        let input = TimeInput {
            time_us: 300_000,
            inc_us: 5_000,
            ply: 20,
            ..base()
        };
        let tm = init(&input);

        // remain_time = 300000 + 0 - 1120 = 298880 (inc is NOT folded in).
        // time_forfeit = false (inc != 0).
        // move_horizon = 160 + 20 - min(20,80) = 160.
        // MTG = min(100000 - 20 + 2, 160)/2 = 160/2 = 80.
        // minimum_time = 1880.
        // remain_estimate = 300000 + 5000*80 + 0 = 700000 - (80+1)*1000 = 619000.
        // t1 = 1880 + 619000/80 = 1880 + 7737 = 9617.
        // t2 = 1880 + (int)(619000*5/80) = 1880 + 38687 = 40567;
        //    cap (int)(619000*0.3)=185700 -> 40567.
        // optimum = min(9617, 298880) = 9617.
        // maximum = min(round_up(40567), 298880):
        //   round_up: ((40567+999)/1000)*1000=41000; -120 = 40880; 40880<40567? no.
        // minimum = min(round_up(1880), 298880) = 1880.
        assert_eq!(tm.minimum(), 1880);
        assert_eq!(tm.optimum(), 9617);
        assert_eq!(tm.maximum(), 40880);
    }

    #[test]
    fn increment_is_excluded_from_remain_time() {
        // The Fischer increment is credited only *after* the move is played, so
        // it must not enter `remain_time` (`timeman.cpp` at the pin).
        // A tiny main clock with a huge increment makes the difference visible:
        // `remain_time` binds both optimum and maximum, and it is computed from
        // the main clock alone.
        let input = TimeInput {
            time_us: 5_000,
            inc_us: 60_000,
            ply: 1,
            ..base()
        };
        let tm = init(&input);

        // remain_time = 5000 + 0 - 1120 = 3880 (with inc it would be 63880).
        // move_horizon = 160 + 20 - min(1,80) = 179; MTG = 179/2 = 89.
        // minimum_time = 1880.
        // remain_estimate = 5000 + 60000*89 - (89+1)*1000 = 5255000.
        // t1 = 1880 + 5255000/89 = 60924; t2 is larger still.
        // optimum = min(t1, 3880) = 3880; maximum = min(round_up(t2), 3880) = 3880.
        // minimum = min(round_up(1880), 3880) = 1880.
        assert_eq!(tm.minimum(), 1880);
        assert_eq!(tm.optimum(), 3880);
        assert_eq!(tm.maximum(), 3880);
    }

    #[test]
    fn sudden_death_ratio_uncapped_above_five_minutes() {
        // Sudden death, time >= 5 min: max_ratio stays 5.0.
        let input = TimeInput {
            time_us: 600_000,
            ply: 1,
            ..base()
        };
        let tm = init(&input);

        // remain_time = 600000 - 1120 = 598880.
        // time_forfeit = true. move_horizon = 160 + 40 - min(1,40) = 199.
        // MTG = min(100000 - 1 + 2, 199)/2 = 199/2 = 99.
        // minimum_time = 1880.
        // remain_estimate = 600000 + 0 + 0 = 600000 - (99+1)*1000 = 500000.
        // t1 = 1880 + 500000/99 = 1880 + 5050 = 6930.
        // max_ratio = min(5.0, max(600000/60000, 1.0)) = min(5.0, 10.0) = 5.0.
        // t2 = 1880 + (int)(500000*5/99) = 1880 + 25252 = 27132;
        //    cap (int)(500000*0.3)=150000 -> 27132.
        // optimum = 6930.
        // maximum = round_up(27132): ((27132+999)/1000)*1000=28000; -120=27880.
        assert_eq!(tm.optimum(), 6930);
        assert_eq!(tm.maximum(), 27880);
    }

    #[test]
    fn sudden_death_ratio_clamped_under_one_minute() {
        // Sudden death, time < 1 min: max_ratio clamps to 1.0.
        let input = TimeInput {
            time_us: 30_000,
            ply: 1,
            ..base()
        };
        let tm = init(&input);

        // remain_time = 30000 - 1120 = 28880.
        // time_forfeit = true. move_horizon = 199. MTG = 99.
        // minimum_time = 1880.
        // remain_estimate = 30000 - 100000 = -70000 -> max(0) = 0.
        // t1 = 1880 + 0 = 1880.
        // max_ratio = min(5.0, max(30000/60000=0.5, 1.0)) = 1.0.
        // t2 = 1880 + (int)(0*1/99) = 1880; cap (int)(0*0.3)=0 -> min(1880,0)=0.
        // optimum = min(1880, 28880) = 1880.
        // maximum = min(round_up(0), 28880):
        //   round_up(0): ((0+999)/1000)*1000=0; max(0,2000)=2000; 2000-120=1880;
        //   1880 < 0? no; min(1880, 28880)=1880.
        // minimum = min(round_up(1880), 28880) = 1880.
        assert_eq!(tm.minimum(), 1880);
        assert_eq!(tm.optimum(), 1880);
        assert_eq!(tm.maximum(), 1880);
    }

    #[test]
    fn final_push_when_time_under_byoyomi_1_2() {
        // Byoyomi 1000, main clock 500 < 1000*1.2 = 1200: final push spends
        // byoyomi + time, and isFinalPush is set.
        let input = TimeInput {
            time_us: 500,
            byoyomi_us: 1_000,
            ply: 50,
            ..base()
        };
        let tm = init(&input);

        // remain_time = 500 + 1000 - 1120 = 380 (>= 100).
        // final push: 500 < (int)(1000*1.2)=1200 -> all = 1000 + 500 = 1500,
        //   isFinalPush = true.
        // then minimum = min(round_up(1500), 380):
        //   round_up(1500): ((1500+999)/1000)*1000=2000; max(2000,2000)=2000;
        //   2000-120=1880; 1880<1500? no; min(1880, 380)=380.
        // optimum = min(1500, 380) = 380.
        // maximum = min(round_up(1500), 380) = 380.
        assert!(tm.is_final_push);
        assert_eq!(tm.minimum(), 380);
        assert_eq!(tm.optimum(), 380);
        assert_eq!(tm.maximum(), 380);
    }

    #[test]
    fn mtg_equal_one_spends_remaining() {
        // At the draw horizon (MTG == 1) all three times equal remain_time.
        // max_moves_to_draw - ply + 2 == 2 gives MTG = 1.
        let input = TimeInput {
            time_us: 60_000,
            byoyomi_us: 5_000,
            max_moves_to_draw: 100,
            ply: 100,
            ..base()
        };
        let tm = init(&input);
        // remain_time = 60000 + 5000 - 1120 = 63880.
        // MTG = min(100 - 100 + 2, move_horizon)/2 = min(2, ..)/2 = 1.
        assert_eq!(tm.minimum(), 63880);
        assert_eq!(tm.optimum(), 63880);
        assert_eq!(tm.maximum(), 63880);
    }

    #[test]
    fn mtg_non_positive_sets_error_and_500() {
        // max_moves_to_draw - ply + 2 <= 0 -> MTG <= 0 error path.
        let input = TimeInput {
            time_us: 60_000,
            max_moves_to_draw: 10,
            ply: 20,
            ..base()
        };
        let tm = init(&input);
        assert!(tm.mtg_error);
        assert_eq!(tm.minimum(), 500);
        assert_eq!(tm.optimum(), 500);
        assert_eq!(tm.maximum(), 500);
    }

    #[test]
    fn round_up_to_fullsecond_false_uses_units_of_one() {
        // With RoundUpToFullSecond off the minimum floor drops to 1 and round_up
        // does not round to a second.
        let input = TimeInput {
            time_us: 300_000,
            inc_us: 5_000,
            ply: 20,
            round_up_to_fullsecond: false,
            ..base()
        };
        let tm = init(&input);

        // remain_time = 298880 (inc excluded; floor is 1, unaffected).
        // minimum_time = max(2000 - 120, 1) = 1880.
        // remain_estimate = 300000 + 5000*80 = 700000 (no -=(MTG+1)*1000).
        // t1 = 1880 + 700000/80 = 1880 + 8750 = 10630.
        // t2 = 1880 + (int)(700000*5/80) = 1880 + 43750 = 45630;
        //    cap (int)(700000*0.3)=210000 -> 45630.
        // optimum = 10630.
        // maximum = round_up(45630) with round off:
        //   max(45630, 2000)=45630; -120 = 45510; min(45510, 298880)=45510.
        // minimum = round_up(1880) off: max(1880,2000)=2000; -120=1880.
        assert_eq!(tm.minimum(), 1880);
        assert_eq!(tm.optimum(), 10630);
        assert_eq!(tm.maximum(), 45510);
    }

    #[test]
    fn slow_mover_scales_optimum_only() {
        let baseline = init(&TimeInput {
            time_us: 600_000,
            byoyomi_us: 10_000,
            ply: 1,
            ..base()
        });
        let slow = init(&TimeInput {
            time_us: 600_000,
            byoyomi_us: 10_000,
            ply: 1,
            slow_mover: 200,
            ..base()
        });
        // optimum doubles (t1.min(remain) * 200/100), maximum unchanged.
        assert_eq!(slow.optimum(), baseline.optimum() * 2);
        assert_eq!(slow.maximum(), baseline.maximum());
    }

    #[test]
    fn usi_ponder_bonus_adds_a_quarter() {
        let off = init(&TimeInput {
            time_us: 600_000,
            byoyomi_us: 10_000,
            ply: 1,
            ..base()
        });
        let on = init(&TimeInput {
            time_us: 600_000,
            byoyomi_us: 10_000,
            ply: 1,
            usi_ponder: true,
            ..base()
        });
        // The bonus is applied to optimumTime pre-clamp; here the clamp does not
        // bind, so it is exactly optimum + optimum/4.
        assert_eq!(on.optimum(), off.optimum() + off.optimum() / 4);
    }

    #[test]
    fn usi_ponder_bonus_suppressed_by_stochastic_ponder() {
        let plain = init(&TimeInput {
            time_us: 600_000,
            byoyomi_us: 10_000,
            ply: 1,
            ..base()
        });
        let both = init(&TimeInput {
            time_us: 600_000,
            byoyomi_us: 10_000,
            ply: 1,
            usi_ponder: true,
            stochastic_ponder: true,
            ..base()
        });
        assert_eq!(both.optimum(), plain.optimum());
    }

    #[test]
    fn rtime_result_within_bounds() {
        // rtime r plus a decaying random increment in [0, min(r/2, r*10/ply)).
        // ply 40 -> bound = min(2000, 1000) = 1000, so result in [4000, 5000).
        let r: i64 = 4000;
        let ply = 40;
        for seed in 1..50u64 {
            let mut prng = Prng::new(seed);
            let tm = TimeManagement::init(
                &TimeInput {
                    rtime: r,
                    ply,
                    ..base()
                },
                &mut prng,
            );
            assert!(
                tm.optimum() >= r && tm.optimum() < r + 1000,
                "rtime out of bounds: {} not in [{}, {})",
                tm.optimum(),
                r,
                r + 1000
            );
            assert_eq!(tm.minimum(), tm.optimum());
            assert_eq!(tm.maximum(), tm.optimum());
        }
    }

    #[test]
    fn round_up_boundaries_full_second_branch() {
        // A movetime init leaves round_up parameters at the defaults
        // (minimum_thinking_time 2000, network_delay 120, remain_time = movetime).
        // Use a large remain_time so the cap does not bind.
        let tm = init(&TimeInput {
            movetime: 1_000_000,
            ..base()
        });
        // Exactly on a whole second: 3000 -> ((3000+999)/1000)*1000 = 3000;
        // max(3000,2000)=3000; -120 = 2880; 2880 < 3000 -> +1000 = 3880.
        assert_eq!(tm.round_up(3000), 3880);
        // Just over: 3001 -> ceil to 4000; -120 = 3880; 3880 >= 3001 -> 3880.
        assert_eq!(tm.round_up(3001), 3880);
        // Below the minimum floor: 100 -> ceil 1000; max(1000,2000)=2000;
        // -120 = 1880; 1880 >= 100 -> 1880.
        assert_eq!(tm.round_up(100), 1880);
    }

    #[test]
    fn round_up_no_round_branch() {
        let tm = init(&TimeInput {
            movetime: 1_000_000,
            round_up_to_fullsecond: false,
            ..base()
        });
        // max(3001, 2000)=3001; -120 = 2881.
        assert_eq!(tm.round_up(3001), 2881);
        // max(100, 2000)=2000; -120 = 1880.
        assert_eq!(tm.round_up(100), 1880);
    }

    #[test]
    fn set_search_end_no_ponder_rounds_elapsed() {
        // Without ponder search_end = round_up(max(elapsed, minimum())).
        let mut tm = init(&TimeInput {
            time_us: 600_000,
            byoyomi_us: 10_000,
            ply: 1,
            ..base()
        });
        // minimum() == 1880 here. Elapsed 5000 > minimum, so
        // search_end = round_up(5000): ceil 5000; -120 = 4880; 4880<5000 -> 5880.
        tm.set_search_end(5000);
        assert_eq!(tm.search_end, 5880);

        // Elapsed below minimum uses minimum(): max(500, 1880) = 1880;
        // round_up(1880) = 1880.
        tm.search_end = 0;
        tm.set_search_end(500);
        assert_eq!(tm.search_end, 1880);
    }

    #[test]
    fn set_search_end_final_push_uses_minimum_directly() {
        let mut tm = init(&TimeInput {
            time_us: 500,
            byoyomi_us: 1_000,
            ply: 50,
            ..base()
        });
        assert!(tm.is_final_push);
        // minimum() == 380 (clamped to remain_time). Elapsed 100 < minimum:
        // t2 = minimum() = 380 (final push branch); max(100, 380) = 380;
        // round_up(380): ceil 1000; max(1000,2000)=2000; -120=1880; 1880<380? no;
        // min(1880, remain_time=380) = 380.
        tm.set_search_end(100);
        assert_eq!(tm.search_end, 380);
    }
}
