function img_with_tones(img, tones, val=100)
    img = copy(img)
    for tone in tones
        add_tone_to_img!(img, tone; val=val)
    end
    return img
end
  
function add_tone_to_img!(img, tone; val=100)
    (region, erb) = tone
    yval = Int(floor(AudioInference.pos_for_erb_val(erb)))
    for i = region[1]:region[2]
        img[yval, i] = val
    end
end

function add_vert_bars(img, xvals; val=100)
    img = copy(img)
    for xval in xvals
        add_vert_bar!(img, xval, val=val)
    end
    img
end
function add_vert_bar!(img, xval; val=100)
    img[:, xval] .= val
end

function draw_rect!(img, ((x1, y1), (x2, y2)); color=100)
    for x = x1:min(x2, size(img)[1])
        img[x, min(y1, size(img)[2])] = color
        img[x, min(y2, size(img)[2])] = color
    end
    for y = y1:min(y2, size(img)[2])
        img[min(x1, size(img)[1]), y] = color
        img[min(x2, size(img)[1]), y] = color
    end
end
function img_with_rects(img, rects; color=100)
    img = copy(img)
    for rect in rects
        draw_rect!(img, rect; color=color)
    end
    return img
end

function add_vert_segs(img, segs; val=100)
    img = copy(img)
    for seg in segs
        add_vert_seg!(img, seg, val=val)
    end
    img
end
function add_vert_seg!(img, seg; val=100)
    for y = seg.ymin:min(seg.ymax, size(img)[1])
        img[y, seg.x] = val
    end
end

function img_with_sources(img, sources, scenelength; val=100)
    img = copy(img)
    for s in sources
        add_source_to_img!(img, s, scenelength)
    end
    img
end
function add_source_to_img!(img, source, scenelength; val=100)
    if source.is_noise
        add_noise_to_img!(img, source, scenelength)
    else
        add_tone_to_img!(img, source, scenelength)
    end
end
function add_tone_to_img!(img, source, scenelength; val=100)
    y = Int(floor(Detector.pos_for_erb_val(source.amp_or_erb)))
    xwidth = size(img)[2]
    start = Int(floor(source.onset / scenelength * xwidth))
    dur = Int(floor(source.duration / scenelength * xwidth))
    for x = start:min(start + dur, xwidth)
        img[y, x] = val
    end
end
function add_noise_to_img!(img, source, scenelength; val=100)
    xwidth = size(img)[2]
    start = Int(floor(source.onset / scenelength * xwidth))
    dur = Int(floor(source.duration / scenelength * xwidth))
    draw_rect!(img, ((1, start), (size(img)[1], min(start + dur, xwidth))); color=val)
end

function rect_for_source(img, source, scenelength)
    TONESIZE = 10
    xwidth = size(img)[2]
    start = Int(floor(source.onset / scenelength * xwidth))
    dur = Int(floor(source.duration / scenelength * xwidth))
    xmin, xmax = start, min(start + dur, xwidth)
    if !source.is_noise
        y = Int(floor(Detector.pos_for_erb_val(source.amp_or_erb)))
        ymin, ymax = max(1, y - Int(TONESIZE / 2)), min(size(img)[1], y + Int(TONESIZE / 2))
    else
        ymin, ymax = 1, size(img)[1]
    end
    return ((ymin, xmin), (ymax, xmax))
end

function img_with_source_rects(img, sources, scenelength; val=100)
    img = copy(img)
    for s in sources
        draw_rect!(img, rect_for_source(img, s, scenelength); color=val)
    end
    img
end