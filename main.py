from range import Range
import numpy as np
import plotly.express as px
import pandas as pd


NB_RANGES = 10000
NB_TESTS = 100
MAX_RANGE_VAL = 1000
NB_BUCKETS = 10


def real_overlap(rand_ranges: np.ndarray, const_ranges):
    nb_line = np.zeros(len(const_ranges))
    for idx, cr in enumerate(const_ranges):
        n = 0
        for rr in rand_ranges:
            if range_overlaps(cr, rr):
                n += 1
        nb_line[idx] = n
    return nb_line


def range_overlaps(r1, r2):
    return r1.start < r2.start < r1.end or\
           r1.start < r2.end < r1.end or\
           r2.start < r1.start < r2.end or\
           r2.start < r1.end < r2.end


def analyze_norm_lin_approx(hist, const_ranges, bins):
    nb_line = np.zeros(len(const_ranges))

    for idx, r in enumerate(const_ranges):
        end_idx, start_idx = bound_idx(bins, r)
        if end_idx >= len(hist):
            # If the constant range goes to far right
            end_idx = len(hist) - 1

        if start_idx == end_idx:
            length = r.end - r.start
            nb_line[idx] += length / (bins[1] - bins[0]) * hist[start_idx]
            continue

        len_start = bins[start_idx+1] - r.start
        nb_line[idx] += len_start / (bins[1] - bins[0]) * hist[start_idx]
        len_end = r.end - bins[end_idx]
        nb_line[idx] += len_end / (bins[1] - bins[0]) * hist[end_idx]

        if start_idx != end_idx - 1:
            # We got intermediate bins
            for i in range(start_idx + 1, end_idx):
                nb_line[idx] += hist[i]
    return nb_line


def analyze_norm_no_approx(hist, const_ranges, bins):
    nb_line = np.zeros(len(const_ranges))

    for idx, r in enumerate(const_ranges):
        end_idx, start_idx = bound_idx(bins, r)
        for i in range(start_idx, end_idx + 1):
            if i >= len(hist):
                break
            nb_line[idx] += hist[i]
    return nb_line


def analyze_hist(hist, const_ranges, bins, lin_approx=False):
    nb_line = analyze_norm_lin_approx(hist, const_ranges, bins) if lin_approx \
        else analyze_norm_no_approx(hist, const_ranges, bins)
    return nb_line


def build_histos(rand_ranges):
    histo_mean_norm = np.zeros(NB_BUCKETS)
    histo_local_norm = np.zeros(NB_BUCKETS)
    histo_up_bound = np.zeros(NB_BUCKETS)
    histo_low_bound = np.zeros(NB_BUCKETS)

    max_val = rand_ranges.max().end
    min_val = rand_ranges.min().start
    print("extrema :", min_val, max_val)
    bin_step = (max_val - min_val) / NB_BUCKETS
    bins = np.array(
        [min_val + i * bin_step for i in range(NB_BUCKETS + 1)]
    )
    print("bins :", bins)

    n_tot = 0
    for r in rand_ranges:
        # Compute the number of bins which this range overlaps
        end_idx, start_idx = bound_idx(bins, r)

        histo_up_bound[end_idx] += 1
        histo_low_bound[start_idx] += 1  # Todo : return
        n = end_idx - start_idx + 1  # Nb of buckets this range overlaps
        n_tot += n

        for i in range(start_idx, end_idx + 1):
            histo_local_norm[i] += 1 / n
            histo_mean_norm[i] += 1

    # TODO : cumuler les histogrammes de bounds
    # Normalize the mean norm histogram
    histo_mean_norm /= n_tot / NB_RANGES
    # print("mean norm :", histo_mean_norm)
    # print("local norm :", histo_local_norm)

    return histo_local_norm, histo_mean_norm, bins


def bound_idx(bins, r):
    idx = 0
    while idx < len(bins) - 1 and bins[idx + 1] < r.start:
        # Haven't reach the first bin of this range
        idx += 1
    start_idx = idx

    while idx < len(bins) - 1 and bins[idx + 1] < r.end:
        # Haven't reach the last bin of this range
        idx += 1
    end_idx = idx
    return end_idx, start_idx


def main():
    # Generate some random ranges to simulate a database range content
    rand_ranges = [Range(MAX_RANGE_VAL) for _ in range(NB_RANGES)]
    rand_ranges = np.array(rand_ranges)

    # Constant ranges to be compared with
    # ex : serange_range && const_range[i]
    const_ranges = [Range(MAX_RANGE_VAL) for _ in range(NB_TESTS)]
    const_ranges = np.array(const_ranges)
    const_ranges_len = [r.len() for r in const_ranges]

    # Range_typanalyze
    histo_local_norm, histo_mean_norm, bins = build_histos(rand_ranges)

    print("Histograms are built")

    real_line_nb = real_overlap(rand_ranges, const_ranges)

    # Estimate
    res_mean_norm = analyze_hist(histo_mean_norm, const_ranges, bins)
    res_mean_norm_approx = analyze_hist(histo_mean_norm, const_ranges, bins, True)
    res_loc_norm = analyze_hist(histo_local_norm, const_ranges, bins, False)

    data = {
        "Ref ranges": const_ranges,
        "Ref length": const_ranges_len,
        "Real nb of &&": real_line_nb,
        "Mean norm. estimation": res_mean_norm,
        "Mean norm. app lin": res_mean_norm_approx,
        "Delta mean-real": res_mean_norm - real_line_nb,
        "Local norm. estimation": res_loc_norm,
        "Delta loc-real": res_loc_norm - real_line_nb
    }

    df = pd.DataFrame(data)

    print(df)
    fig = px.scatter(df, y="Delta mean-real", x="Ref length")
    fig.show()


if __name__ == '__main__':
    main()
