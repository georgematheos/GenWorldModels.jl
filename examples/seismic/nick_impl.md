# Notes on Nick Hay's example inference program

I'm looking through the code Nick wrote for the seismic monitoring inference, and am
going to take some notes to make sure I understand the simplifications, etc., that
he is using.

### Model
```
num_events ~ poisson(TOTAL_EVENT_INTENSITY)
(mem) event_time(ev) ~ uniform(TIME[0], TIME[1])
(mem) event_loc(ev) ~ uniform(SPACE[0], SPACE[1])
(mem) event_mag ~ EVENT_MAG_MIN + exponential(1/log(10))
(mem) num_noise(det) ~ poisson(TOTAL_NOISE_INTENSITY)

fixed obs time, obs amp, STD

true_events ~ [~event_s() for i=1:num_events]

for each detector:
  `event_blips_assoc_s(detector_id, true_events)`

for each detector:
  `blip[detector_id].extend(noise_blips_s(detector_id))`
```

Looks like the travel velocity is fixed to exactly 1.
Fixed `MAG_DECAY`.

### Potential blips
For detector (d1, d2):
    For blip pair (b1, b2) sufficiently close to each other in time
        (sufficiently close related to dist between stations)       :


## Simplifications, and how I've replicated them

Simplifications I have replicated exactly by constraining the values of samples
in the full model:
1. Constant event occurance rate.
2. Constant velocity (=1.)
3. Constant absorpivity per unit distance (MAG decay)
4. Constant false alarm rate shared by all stations.
5. Constant time measurement error parameters (shared by all stations)
6. Constant amplitude measurement error parameters (shared by all stations)

Simplifications I have not replicated exactly:
1. All stations detect all events with probability 1.  To approximate this, I have fixed the parameters of all the priors on how likely it is to detect an event so it is highly likely each event is detected.
2. Noise measurement amplitudes are generated from an exponential with fixed min, rate.  I have fixed the normal parameters for each station to have the same 1st and 2nd moments as this exponential, but have not changed the distribution family.