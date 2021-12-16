from range import Range
import numpy as np
import plotly.express as px
import pandas as pd
import statistics
import random

NB_RANGES = 100000
NB_TESTS = 100
MAX_RANGE_VAL = 10000
NB_BUCKETS = 100
# We keep the same amount of buckets for the second relationship but we decrease the max value, the size of bins will
# decrease too.
MAX_RANGE_VAL_2 = 10000


def real_overlap(rand_ranges: np.ndarray, const_ranges):
    nb_line = np.zeros(len(const_ranges))
    for idx, cr in enumerate(const_ranges):
        n = 0
        for rr in rand_ranges:
            if range_overlaps(cr, rr):
                n += 1
        nb_line[idx] = n
    return nb_line


def real_join(rand_ranges: np.ndarray, rand_ranges_2: np.ndarray):
    """
    Method that determines the real number of overlaps between to relationships.
    :param rand_ranges: Ranges of the first relationship.
    :param rand_ranges_2: Ranges of the second relationship.
    :return: The number of lines that are overlapping.
    """
    nb_line = 0
    for r in rand_ranges:
        for r2 in rand_ranges_2:
            if range_overlaps(r, r2):
                nb_line += 1
    return nb_line


def range_overlaps(r1, r2):
    return r1.start < r2.start < r1.end or \
           r1.start < r2.end < r1.end or \
           r2.start < r1.start < r2.end or \
           r2.start < r1.end < r2.end


def analyze_new_lines(hist_appart, hist_new_lines, const_ranges, bins):
    """
    Sums the contribution of every bucket with the new lines histogram values except for the
    first bucket (the one where the const_range begins).
    :param hist_appart:
    :param hist_new_lines:
    :param const_ranges:
    :param bins:
    :return:
    """
    nb_line = np.zeros(len(const_ranges))

    for idx, r in enumerate(const_ranges):
        endidx, startidx = bound_idx(bins, r)
        nb_line[idx] += hist_appart[startidx]  # On ajoute comme première valeur, le nombre de lignes qui apparaissent
        # dans le bucket de départ.
        for i in range(startidx + 1, endidx + 1):
            if i >= len(nb_line):
                break
            nb_line[idx] += hist_new_lines[i]
    return nb_line


def analyze_new_lines_lin_approx(hist_appart, hist_new_lines, const_ranges, bins):
    """
    Same as analyze_lin_approx but with a linear approximation at the beginning.
    :param hist_appart:
    :param hist_new_lines:
    :param const_ranges:
    :param bins:
    :return:
    """
    nb_line = np.zeros(NB_TESTS)
    length_bin = bins[1]-bins[0]

    for idx, r in enumerate(const_ranges):
        endidx, startidx = bound_idx(bins, r)


        # Starting bucket
        piece_r_in_bin = bins[startidx+1] - r.start
        presence_rate = piece_r_in_bin/length_bin
        nb_line[idx] += hist_appart[startidx]#*presence_rate  # On ajoute comme première valeur, le nombre de lignes qui apparaissent
        # dans le bucket de départ.

        # Ending bucket
        piece_r_in_bin = r.end - bins[endidx]
        presence_rate = piece_r_in_bin/length_bin
        nb_line[idx] += hist_new_lines[startidx]*presence_rate
        for i in range(startidx + 1, endidx):
            if i >= NB_BUCKETS:
                break
            nb_line[idx] += hist_new_lines[i]
    return nb_line


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

        len_start = bins[start_idx + 1] - r.start
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


def analyze_join_hists(hist, hist_2, bins, bins_2):
    """
    For each hist bucket, multiply the value of that bucket with the value of
    the hist2 bucket(s) it overlaps.
    :param hist: Histogram of new lines for the first relationship
    :param hist_2: Histogram of "appartenance" for the second relationship
    :param bins: Values at the limits of the buckets of the hist
    :param bins_2: Values at the limits of the buckets of the hist_2
    :return: The cardinality of the join between the two relationships.
    """
    nb_lines = 0

    for i in range(len(hist)):
        idx_buckets_2 = buckets_overlapped(bins, bins_2, i)
        for idx in idx_buckets_2:
            nb_lines += hist[i] * hist_2[idx]
    return nb_lines


def analyze_hist(hist, const_ranges, bins, lin_approx=False):
    nb_line = analyze_norm_lin_approx(hist, const_ranges, bins) if lin_approx \
        else analyze_norm_no_approx(hist, const_ranges, bins)
    return nb_line


def analyze_hist_new_lines(hist_appart, hist_new_lines, const_ranges, bins, lin_approx=False):
    nb_line = analyze_new_lines_lin_approx(hist_appart, hist_new_lines, const_ranges, bins) if lin_approx \
        else analyze_new_lines(hist_appart, hist_new_lines, const_ranges, bins)
    return nb_line


def build_appartenance_hist(rand_ranges, bins):
    histo = np.zeros(NB_BUCKETS)

    n = 0
    for r in rand_ranges:
        # Compute the number of bins which this range overlaps
        end_idx, start_idx = bound_idx(bins, r)
        n += end_idx - start_idx + 1  # Nb of buckets this range overlaps

        for i in range(start_idx, end_idx + 1):
            if i >= len(histo):
                break
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
            if i >= len(histo):
                break
            histo[i] += 1 / n
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
    # print("bins :", bins)

    histo_appart, mean_nb_overlap = build_appartenance_hist(rand_ranges, bins)
    histo_mean_norm = np.copy(histo_appart) / mean_nb_overlap
    histo_local_norm = build_local_norm(rand_ranges, bins)
    histo_low_bound, histo_up_bound = build_bound_hists(rand_ranges, bins)

    return histo_local_norm, histo_mean_norm, histo_low_bound, histo_up_bound, histo_appart, bins


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

    while idx +1 < len(bins) - 1 and bins[idx + 1] < r.end:
        # Haven't reach the last bin of this range
        idx += 1
    end_idx = idx
    return end_idx, start_idx


def buckets_overlapped(bins, bins_2, i):
    """
    Allows you to define which buckets of relationship 2, relationship 1 overlaps.
    :param bins: Values at the limits of the buckets of the hist
    :param bins_2: Values at the limits of the buckets of the hist_2
    :param i: Index of the bucket in the hist
    :return: List of indexes of buckets that are overlapping the bucket in the hist.
    """
    idx_overlap = []
    # print("Index du bucket dans A :", i)
    startidx = bins[i]
    endidx = bins[i + 1]
    for j in range(len(bins_2) - 1):
        if (startidx == bins_2[j]) or (endidx > bins_2[j] > startidx) or (bins_2[j + 1] > startidx > bins_2[j]):
            idx_overlap.append(j)
            # print("Overlap avec le bucket ", j, " dans B.")
        elif bins_2[j] >= endidx:
            break
    return idx_overlap


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
    histo_local_norm, histo_mean_norm, histo_low, histo_up, hist_appart, bins = build_histos(rand_ranges)

    print("Histograms are built")

    real_line_nb = real_overlap(rand_ranges, const_ranges) / NB_RANGES

    print("The real number of lines is calculated")

    # Estimate
    res_mean_norm = analyze_hist(histo_mean_norm, const_ranges, bins) / NB_RANGES
    res_mean_norm_approx = analyze_hist(histo_mean_norm, const_ranges, bins, True) / NB_RANGES
    res_loc_norm = analyze_hist(histo_local_norm, const_ranges, bins, False) / NB_RANGES
    res_new_lines = analyze_hist_new_lines(hist_appart, histo_low, const_ranges, bins) / NB_RANGES
    res_new_lines_lin_approx = analyze_hist_new_lines(hist_appart, histo_low, const_ranges, bins, True) / NB_RANGES

    # To test the accuracy of our method for join cardinality, we generate many second relationships
    # real_line_join_nb, res_join_cardinality, second_rel_length = test_join_cardinality(bins, rand_ranges)

    cols = ["Ref range idx", "Real fraction", "Selectivity", "Delta with real", "Type of approx"]
    types_estimation = ["Mean normalization", "Mean normalization with approx", "Local normalization",
                        "Counting method", "Counting method with approach", "Real fraction"]
    datafr = pd.DataFrame(columns=cols)

    #datafr = add_to_dataframe(datafr, real_line_nb, res_mean_norm, types_estimation[0])
    # datafr = add_to_dataframe(datafr, real_line_nb, res_mean_norm_approx, types_estimation[1])
    #datafr = add_to_dataframe(datafr, real_line_nb, res_loc_norm, types_estimation[2])
    datafr = add_to_dataframe(datafr, real_line_nb, res_new_lines, types_estimation[3])
    # datafr = add_to_dataframe(datafr, real_line_nb, res_new_lines_lin_approx, types_estimation[4])
    # datafr = add_to_dataframe(datafr, real_line_nb, real_line_nb, types_estimation[5])

    print(datafr)
    print("Mean of difference is :", statistics.mean(res_new_lines - real_line_nb))
    print("Std dev of diff is :", statistics.stdev(res_new_lines - real_line_nb))

    fig = px.scatter(datafr, y=cols[3], x=cols[1],
                     title="Difference between the estimated and the real selectivity ")
    # fig = px.scatter(datafr, y=cols[3], x=cols[1], color=cols[4],
    #                  title="Difference between the estimated and the real selectivity of the different estimation "
    #                       "methods")
    fig.show()


def add_to_dataframe(datafr, real_line_nb, estimation_values, type_of_appr: str):
    datafr_app = pd.DataFrame({
        'Ref range idx': np.arange(0, NB_TESTS),
        'Real fraction': real_line_nb,
        'Selectivity': estimation_values,
        'Delta with real': estimation_values - real_line_nb,
        'Type of approx': type_of_appr
    })
    return datafr.append(datafr_app, ignore_index=True)


def test_join_cardinality(bins, rand_ranges):
    res_join_cardinality = np.zeros(NB_TESTS)
    real_line_join_nb = np.zeros(NB_TESTS)
    second_rel_length = np.zeros(NB_TESTS)
    print("Generating second table for join")
    for i in range(NB_TESTS):
        # Randomly generate the length of the second relationship
        length = random.randint(400, MAX_RANGE_VAL_2)
        second_rel_length[i] = length
        # Generate another set of randoms range to test the calculation of join cardinality
        rand_ranges_2 = [Range(length) for _ in range(NB_RANGES)]
        rand_ranges_2 = np.array(rand_ranges_2)
        # Construction of histograms for the join cardinality
        histo_local_norm_2, histo_mean_norm_2, histo_low_2, histo_up_2, hist_appart_2, bins_2 = \
            build_histos(rand_ranges_2)
        real_line_join_nb[i] = real_join(rand_ranges, rand_ranges_2)
        res_join_cardinality[i] = analyze_join_hists(histo_low_2, hist_appart_2, bins, bins_2)
    return real_line_join_nb, res_join_cardinality, second_rel_length


if __name__ == '__main__':
    main()
