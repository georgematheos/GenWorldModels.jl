module Detector
using DSP: conv
using ImageMorphology: label_components, component_boxes
using Memoize

const MIN_SOURCE_LENGTH = 10

#######################################
# Sound detection as image processing #
#######################################

const SOBEL_FILTER = [
    -0.2 -0.2 -0.2 -0.5 0 0.5 0.2 0.2 0.2;
    -0.2 -0.2 -0.2 -0.5 0 0.5 0.2 0.2 0.2;
    -0.2 -0.2 -0.2 -0.5 0 0.5 0.2 0.2 0.2;
    -0.4  -0.4 -0.4 -1.0 0 1.0 0.4 0.4 0.4;
    -0.2 -0.2 -0.2 -0.5 0 0.5 0.2 0.2 0.2;
    -0.2 -0.2 -0.2 -0.5 0 0.5 0.2 0.2 0.2;
    -0.2 -0.2 -0.2 -0.5 0 0.5 0.2 0.2 0.2;
]

# [
#     -0.2 -0.5 0 0.5 0.2;
#     -0.4 -1.0 0 1.0 0.4;
#     -0.2 -0.5 0 0.5 0.2
# ]

"""
    VertSegment(x, ymin, ymax)

Represents a vertical line segment in a 2D array.
"""
struct VertSegment
    x::Int
    ymin::Int
    ymax::Int
end
startseg(((ymin, xmin), (ymax, xmax))) = VertSegment(xmin, ymin, ymax)
seglength(s::VertSegment) = s.ymax - s.ymin

"""
    get_start_end_segs(gram; threshold=0.3)

Finds `VertSegment`s corresponding to the starts and ends of sounds.
Does this using a Sobel edge detector.
`threshold` specifies the minimum fraction of the maximum-intensity-pixel's intensity
in the filtered image which a pixel must pass to be counted as a detection of an edge.
"""
function get_start_end_segs(gram; threshold=0.3)
    # apply edge-detector filter which produces vertical bars of negative values
    # at sound-intensity increases, and bars of positive values at sound-intensity decreases
    edge_img = try
        conv(gram, SOBEL_FILTER)
    catch
        println("gram: $gram")
        error()
    end

    # println("edge img max: ", maximum(edge_img))
    # println("edge img min: ", minimum(edge_img))

    start_img = edge_img .< minimum(edge_img)*threshold
    end_img = edge_img .> maximum(edge_img)*(threshold * 3/5)

    # display(start_img)

    startsegs = map(startseg, (component_boxes ∘ label_components)(start_img)[2:end])
    endsegs = map(startseg, (component_boxes ∘ label_components)(end_img)[2:end])

    return (startsegs, endsegs)
end

"""
    possible_noise_tone_starts(startsegs, maxy)

Given a collection of segments starting sounds, separates these into which could be starts
of tones vs starts of noises according to a heuristic.  Returns lists `(noisesegs, tonesegs)`.
"""
function possible_noise_tone_starts(startsegs, maxy)
  noisesegs = []
  tonesegs = []
  for seg in startsegs
    if could_be_noise(seg, maxy)
      push!(noisesegs, seg)
    end
    if could_be_tone(seg, maxy)
      push!(tonesegs, seg)
    end
  end
  return (noisesegs, tonesegs)
end
could_be_noise(seg, maxy) = seg.ymax - seg.ymin > maxy/3
could_be_tone(seg, maxy) = let diff = seg.ymax - seg.ymin; diff >= 2 && diff <= maxy/2; end

"""
    possible_endsegs(startseg, endsegs)

Returns all segments in `endsegs` which could be the ending of `startseg`.
"""
possible_endsegs(startseg, endsegs) = (seg for seg in endsegs if could_end(startseg, seg))
function could_end(startseg, endseg)
    (endseg.x - startseg.x > MIN_SOURCE_LENGTH && 
    overlap((startseg.ymin, startseg.ymax), (endseg.ymin, endseg.ymax)) > 0)
end

"""
    overlap((a, b), (c, d))

The Lebesgue measure of (a, b) ∩ (c, d).
"""
function overlap((a, b), (c, d))
    if a > c
        return overlap((c, d), (a, b))
    end
    if b < c
        return 0
    end
    if d < b
        return (d - c) + 1
    end
    return b - c + 1
end

"""
    tonerect(startseg, endseg, maxy)

Returns a rectangle approximating the position in the gammatonegram of a tone
with the given startseg and endseg.
"""
function tonerect(startseg, endseg, maxy)
    if endseg.ymax - endseg.ymin > 4/5 * maxy
        return ((startseg.ymin, startseg.x), (startseg.ymax, endseg.x))
    end
    ymin = min(startseg.ymin, endseg.ymin)
    ymax = max(startseg.ymax, endseg.ymax)
    return ((ymin, startseg.x), (ymax, endseg.x))
end
"""
    noiserect(startseg, endseg, maxy)

Returns a rectangle approximating the position in the gammatonegram of a noise
with the given startseg and endseg.
"""

function noiserect(startseg, endseg, maxy)
    return ((1, startseg.x), (maxy, endseg.x))
end

segpairs(startsegs, endsegs) = Iterators.flatten(
  ((startseg, endseg) for endseg in possible_endsegs(startseg, endsegs))
  for startseg in startsegs
)

"""
    get_noise_tone_rects(gram; threshold=0.3)

Finds bounding-box rectangles within a given gammatonegram for all noises or tones
detected in it according to a heuristic algorithm.  This algorithm typically
produces many false positive detections.

Returns 2 lists of rectangles, `(noiserects, tonerects)`.  A rectangle is represented
as a tuple `((miny, minx), (maxy, maxx))`.
"""
function get_noise_tone_rects(gram; threshold=0.3)
    maxy = size(gram)[1]
    (startsegs, endsegs) = get_start_end_segs(gram; threshold=threshold)
    (noisesegs, tonesegs) = possible_noise_tone_starts(startsegs, maxy)
    tonerects = [tonerect(st, nd, maxy) for (st, nd) in segpairs(tonesegs, endsegs)]
    noiserects = [noiserect(st, nd, maxy) for (st, nd) in segpairs(noisesegs, endsegs)]

    return (noiserects, tonerects)
end

########################################################
# Convert detected sounds in image to symbolic objects #
########################################################

# I derived these formulae by analyzing the generated
# noises/tones at different amp/erb values:
noisesum_for_amp(amp) = 1054 + 64*amp
noise_amp_for_col(col) = (sum(col) - 1054)/(64)
function pos_for_erb_val(x)
    if x ≤ 5
        9 + 2x
    elseif x ≤ 12
        19 + (31/7)*(x-5)
    elseif x ≤ 15
        50 + 3(x-12)
    else
        60.5
    end
end
function tone_erb_for_val(x)
    if x ≤ 19
        0.5(x-9)
    elseif x ≤ 50
        5. + (7.0/31.0)*(x-19.)
    elseif x ≤ 60
        12. + (1.0/3.0)*(x-50.)
    else
        15.
    end
end

struct Source
    is_noise::Bool
    amp_or_erb::Float64
    onset::Float64
    duration::Float64
end

@memoize Dict function get_detected_sources(img; threshold=0.3, scenelength=1)
    (noiserects, tonerects) = get_noise_tone_rects(img, threshold=threshold)

    return collect(Iterators.flatten((
        (tonesource(rect, img, scenelength) for rect in tonerects),
        (noisesource(rect, img, scenelength) for rect in noiserects)
    )))
end

function tonesource(((miny, minx), (maxy, maxx)), img, scenelength)
    img_xwidth = size(img)[2]
    Source(
        false,
        tone_erb_for_val( (miny + maxy) / 2 ),
        minx/img_xwidth * scenelength,
        (maxx - minx)/img_xwidth * scenelength
    )
end
function noisesource(((miny, minx), (maxy, maxx)), img, scenelength)
    img_xwidth = size(img)[2] 
    avg_col = img[:, minx:min(img_xwidth,maxx)] / (min(img_xwidth,maxx) - minx)
    Source(
        true,
        noise_amp_for_col(avg_col),
        minx/img_xwidth * scenelength,
        (maxx - minx)/img_xwidth * scenelength
    )
end

end # module