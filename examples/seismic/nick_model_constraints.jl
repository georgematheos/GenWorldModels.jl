# natural units, so velocity = 1
SPACE_SIZE = max(SPACE_RANGE) - min(SPACE_RANGE)
TIME_SIZE = max(TIME_RANGE) - min(TIME_RANGE)
TOTAL_EVENT_INTENSITY = EVENT_INTENSITY * TIME_SIZE * SPACE_SIZE
TOTAL_NOISE_INTENSITY = TOTAL_NOISE_INTENSITY * TIME_SIZE
TIME_STD_DEV = 2.
MAG_STD_DEV = 1.
NOISE_AMP_MEAN = log(10)
NOISE_AMP_VAR = (1/log(10))^2
# all constraints for the world
choicemap(
    (:event_occurance_rate => (), TOTAL_EVENT_INTENSITY),
    (:signal_velocity => () => :W_0, 1.),
    (:absorpivity_per_unit_distance => (), (MIN_EVENT_MAG - MAG_STD_DEV)/SPACE_SIZE),
    [
        [
            (:background_noise_level => Station(i), 0.),
            (:false_alarm_rate => Station(i), TOTAL_NOISE_INTENSITY),
            (:ν => Station(i), -4.),
            (:σ2 => Station(i), 1.),
            (:arrival_time_measurement_error_params => Station(i), (0., TIME_STD_DEV)),
            (:amplitude_measurement_error_params => Station(i), (0., MAG_STD_DEV)),
            (:noise_detection_amplitude_params => Station(i), (NOISE_AMP_MEAN, NOISE_AMP_VAR))
        ]...
        for i=1:NUM_STATIONS
    ]...
    # NOTE: Nick is sampling all the noise amplitudes from an exponential,
    # which doesn't match the official spec, and can't be implemented
    # by fixing some samples.
)