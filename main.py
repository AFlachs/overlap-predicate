from range import Range
import numpy as np
import plotly.express as px
import pandas as pd
import random

NB_RANGES = 1000
NB_TESTS = 100
MAX_RANGE_VAL = 1000
NB_BUCKETS = 10
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
        for i in range(startidx + 1, endidx):
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
    nb_line = np.zeros(len(const_ranges))
    length_bin = bins[1]-bins[0]

    for idx, r in enumerate(const_ranges):
        endidx, startidx = bound_idx(bins, r)
        piece_r_in_bin = r.start - bins[startidx]
        presence_rate = piece_r_in_bin/length_bin
        nb_line[idx] += hist_appart[startidx]*presence_rate  # On ajoute comme première valeur, le nombre de lignes qui apparaissent
        # dans le bucket de départ.
        for i in range(startidx + 1, endidx):
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
    histo_up_bound, histo_low_bound = build_bound_hists(rand_ranges, bins)

    # Now that we have collected the statistic we needed we can compute the cumulated histograms of the bound
    cumul_low_bound = np.zeros(NB_BUCKETS)
    cumul_up_bound = np.zeros(NB_BUCKETS)

    cumul_low_bound[0] = histo_low_bound[0]
    cumul_up_bound[0] = histo_up_bound[0]
    for idx in range(1, NB_BUCKETS):
        cumul_low_bound[idx] = cumul_low_bound[idx - 1] + histo_low_bound[idx]
        cumul_up_bound[idx] = cumul_up_bound[idx - 1] + histo_up_bound[idx]

    hist_nb_new_ranges = np.zeros(NB_BUCKETS)
    hist_nb_new_ranges[0] = histo_appart[0]
    for i in range(1, NB_BUCKETS):
        hist_nb_new_ranges[i] = histo_appart[i] - (cumul_low_bound[i - 1] - cumul_up_bound[i - 1])

    return histo_local_norm, histo_mean_norm, histo_low_bound, histo_up_bound, hist_nb_new_ranges, histo_appart, bins


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
    histo_local_norm, histo_mean_norm, histo_low, histo_up, hist_nb_new_ranges, hist_appart, bins = build_histos(
        rand_ranges)

    print("Histograms are built")

    real_line_nb = real_overlap(rand_ranges, const_ranges)

    print("The real number of lines is calculated")

    # Estimate
    res_mean_norm = analyze_hist(histo_mean_norm, const_ranges, bins)
    res_mean_norm_approx = analyze_hist(histo_mean_norm, const_ranges, bins, True)
    res_loc_norm = analyze_hist(histo_local_norm, const_ranges, bins, False)
    res_new_lines = analyze_hist_new_lines(hist_appart, hist_nb_new_ranges, const_ranges, bins)
    res_new_lines_lin_approx = analyze_hist_new_lines(hist_appart, hist_nb_new_ranges, const_ranges, bins, True)

    # To test the accuracy of our method for join cardinality, we generate many second relationships
    res_join_cardinality = np.zeros(NB_TESTS)
    real_line_join_nb = np.zeros(NB_TESTS)
    second_rel_length = np.zeros(NB_TESTS)
    for i in range(NB_TESTS):
        # Randomly generate the length of the second relationship
        length = random.randint(400, 10000)
        second_rel_length[i] = length
        # Generate another set of randoms range to test the calculation of join cardinality
        rand_ranges_2 = [Range(length) for _ in range(NB_RANGES)]
        rand_ranges_2 = np.array(rand_ranges_2)
        # Construction of histograms for the join cardinality
        histo_local_norm_2, histo_mean_norm_2, histo_low_2, histo_up_2, hist_nb_new_ranges_2, hist_appart_2, bins_2 = build_histos(
            rand_ranges_2)
        real_line_join_nb[i] = real_join(rand_ranges, rand_ranges_2)
        res_join_cardinality[i] = analyze_join_hists(hist_nb_new_ranges, hist_appart_2, bins, bins_2) / 10

    data = {
        "Ref ranges": const_ranges,
        "Ref length": const_ranges_len,
        "Real nb of &&": real_line_nb,
        "Mean norm. estimation": res_mean_norm,
        "Mean norm. app lin": res_mean_norm_approx,
        "Delta mean-real": res_mean_norm - real_line_nb,
        "Local norm. estimation": res_loc_norm,
        "Delta loc-real": res_loc_norm - real_line_nb,
        "Real nb of join": real_line_join_nb,
        "Delta estimated join-real": res_join_cardinality - real_line_join_nb,
        "Second rel length": second_rel_length,
        "New lines estimation": res_new_lines,
        "New lines approx estimation": res_new_lines_lin_approx,
        "Delta new lines": res_new_lines - real_line_nb,
        "Delta new lines approx": res_new_lines_lin_approx - real_line_nb,
    }

    df = pd.DataFrame(data)

    # print(df[["Ref length", "Delta mean-real", "Delta loc-real"]])

    fig = px.scatter(df, y="Delta new lines", x="Real nb of &&")
    fig.show()

    print("Nb réel de lignes join :", real_line_join_nb, " Nb lignes estimées :", res_join_cardinality)
    print("Le pourcentage d'erreur est de ", ((res_join_cardinality - real_line_join_nb) / real_line_join_nb) * 100)


if __name__ == '__main__':
    main()
