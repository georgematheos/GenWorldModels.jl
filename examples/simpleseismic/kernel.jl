is_detection_of(e, d) = !is_false_positive(d) && @origin(d)[2] == e
is_dubious(e, ds) = sum(is_detection_of(e, d) for d in ds) == 1

@gen function birth_death_proposal(prev_trace)
    dets, events, stations = [@objects(prev_trace, T)
                        for T in [Detection, Event, Station]]
    
    # decide whether to create or delete an event
    false_positives = filter(is_false_positive, dets)
    dubious_events = filter(e -> is_dubious_event(e, dets), events)
    create_prob = isempty(false_positives) ? 0.0 :
                                    isempty(dubious_events) ? 1.0 : 0.5
    create_event ~ bernoulli(create_prob)

    # make additional choices based on move type
    if create_event
        fp_det ~ uniform_choice(false_positives)
        new_evt_mag ~ normal(@get(prev_trace, reading[fp_det]), 0.2)
        new_evt_idx ~ uniform_choice(range(1, length(events) + 1))
    else
        evt_to_delete ~ uniform_choice(dubious_events)
        detection = findfirst(d -> is_detection_of(evt_to_delete, d), dets)
        num_fps = sum(first(origin(d)) == first(origin(detection))
                                for d in fp_detections)
        fp_det_idx ~ uniform_choice(1, num_fps + 1)
    end
end

@oupm_involution function birth_death_involution()
    @set_backward :create_event to !@proposed(:create_event)

    if @proposed(:create_event)
        # create the new dubious event
        new_event = @create Event with index=@proposed[:new_evt_idx]
        @set magnitude => :event_mag of new_event to @proposed(:new_evt_mag)

        # change the false positive to be a true detection of the new event
        station, = @origin @proposed(:fp_det)
        @change @proposed(:fp_det) to origin (station, new_event) index 1

        # fill in the details of the backward move
        @set_backward :evt_to_delete to new_event
        @set_backward :fp_detection_idx to index @proposed(:fp_detection)
    else
        # delete the dubious event
        dubious_evt = @proposed(:evt_to_delete)
        @delete dubious_evt

        # find detection of this event and change to a false positive
        dets = @objects(Detection)
        det = findfirst(d -> is_detection_of_event(dubious_evt, d), dets)
        station = first(@origin det)
        new_fp = @change det to origin (station,) index @proposed(:fp_det_idx)

        # fill in the details of the backward move
        @set_backward :fp_det to new_fp
        @set_backward :new_evt_mag to @get magintude => :mag of dubious_evt
        @set_backward :new_evt_idx to index @proposed(:evt_to_delete)
    end
end