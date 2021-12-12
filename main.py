from range import Range
import numpy as np
import plotly.express as px
import pandas as pd


NB_RANGES = 1000
NB_TESTS = 10
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
    """
    Sums the contributions of every bucket having a non empty intersection with const_range.
    If the const_range overlaps the bucket completely the contribution is the value of the histogram in this bucket,
    otherwise it is a linear interpolation corresponding to the fraction of the overlap.
    :param hist: Statistic to evaluate
    :param const_ranges: List of reference ranges
    :param bins: Values at the limits of the buckets of the histogram
    :return: List of approximated number of line
    """
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
    """
    Sums the values in the buckets which have a non empty intersection with const_range.
    :param hist: Statistic to evaluate
    :param const_ranges: List of reference ranges
    :param bins: Values at the limits of the buckets of the histogram
    :return: List of approximated number of line
    """
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


def build_appartenance_hist(rand_ranges, bins):
    histo = np.zeros(NB_BUCKETS)

    n = 0
    for r in rand_ranges:
        # Compute the number of bins which this range overlaps
        end_idx, start_idx = bound_idx(bins, r)
        n += end_idx - start_idx + 1  # Nb of buckets this range overlaps

        for i in range(start_idx, end_idx + 1):
            histo[i] += 1
    n /= NB_RANGES
    return histo, n


def build_local_norm(rand_ranges, bins):
    histo = np.zeros(NB_BUCKETS)

    for r in rand_ranges:
        # Compute the number of bins which this range overlaps
        end_idx, start_idx = bound_idx(bins, r)
        n = end_idx - start_idx + 1  # Nb of buckets this range overlaps

        for i in range(start_idx, end_idx + 1):
            histo[i] += 1/n
    return histo


def build_bound_hists(rand_ranges, bins):
    hist_up = np.zeros(NB_BUCKETS)
    hist_low = np.zeros(NB_BUCKETS)
    for r in rand_ranges:
        # Compute the number of bins which this range overlaps
        end_idx, start_idx = bound_idx(bins, r)

        hist_up[end_idx] += 1
        hist_low[start_idx] += 1
    return hist_low, hist_up


def build_histos(rand_ranges):
    """
    Build diverse histograms for the column containing rand_ranges
    :param rand_ranges: Column of range types
    :return: Many histos
    """
    max_val = rand_ranges.max().end
    min_val = rand_ranges.min().start

    bin_step = (max_val - min_val) / NB_BUCKETS
    bins = np.array(
        [min_val + i * bin_step for i in range(NB_BUCKETS + 1)]
    )
    print("bins :", bins)

    histo_appart, mean_nb_overlap = build_appartenance_hist(rand_ranges, bins)
    histo_mean_norm = np.copy(histo_appart) / mean_nb_overlap
    histo_local_norm = build_local_norm(rand_ranges, bins)
    histo_up_bound, histo_low_bound = build_bound_hists(rand_ranges, bins)

    # Now that we have collected the statistic we needed we can compute the cumulated histograms of the bound
    cumul_low_bound = np.zeros(NB_BUCKETS)
    cumul_up_bound = np.zeros(NB_BUCKETS)

    cumul_low_bound[0] = histo_low_bound[0]
    cumul_up_bound[0] = histo_up_bound[0]
    for idx in range(1, NB_BUCKETS):
        cumul_low_bound[idx] = cumul_low_bound[idx-1] + histo_low_bound[idx]
        cumul_up_bound[idx] = cumul_up_bound[idx-1] + histo_up_bound[idx]

    hist_nb_new_ranges = np.zeros(NB_BUCKETS)
    hist_nb_new_ranges[0] = histo_appart[0]
    for i in range(1, NB_BUCKETS):
        hist_nb_new_ranges[i] = histo_appart[i] - (cumul_low_bound[i-1] - cumul_up_bound[i-1])

    return histo_local_norm, histo_mean_norm, histo_low_bound, histo_up_bound, hist_nb_new_ranges, bins


def bound_idx(bins, r):
    """
    Compute the indexes of the bounds of range r in histogram with buckets bins
    :param bins: Limits of the histogram buckets
    :param r: Range to fill in the histogram
    :return: end_idx, start_idx
    """
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
    histo_local_norm, histo_mean_norm, histo_low, histo_up, hist_nb_new_ranges, bins = build_histos(rand_ranges)

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

    print(df[["Ref length", "Delta mean-real", "Delta loc-real"]])

    fig = px.scatter(df, y="Delta mean-real", x="Ref length")
    # fig.show()


if __name__ == '__main__':
    main()
