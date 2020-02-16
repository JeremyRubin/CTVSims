import numpy as np
import math

from matplotlib import pyplot as plt
# Max Block Size (ignoring witness vbyte stuff, should have a linear effect)
MAX_BLOCKWEIGHT = 1000000
# The number of blocks to run the simulator for.
# It can be a bit slow, but effects should be visible within 100 blocks,
# depending on distributions and other parameters.
SIM_BLOCKS = 500

# Time in seconds between expected blocks.
AVG_TIME_BETWEEN_BLOCKS = 10*60

# This sets a global "trace" of how long our blocks took to mine
# the exponential distribution measures time between events
time_between_blocks = np.random.exponential(AVG_TIME_BETWEEN_BLOCKS, SIM_BLOCKS)

# Block Arrival times are the cumsum of the relative time between blocks.
absolute_block_arrivals = np.cumsum(time_between_blocks)



# we have a param that is the average rate of transaction arrival globally
# and for this individual business.
AVG_GLOBAL_TXPS = 20
MY_PAYMENT_REQUESTS_PS = 1
# Assume that transactions show up as a poisson process
# This could be a bad assumption, but it just adds variance
# to the process v.s. a boring constant rate (more chance to observe edge
# behavior).
# N.B.: Because np.poisson rounds the argument as an integer, we multiply inside
# the function rather than outside to get a more precise result from the
# distribution. This also allows us to use non-integer TXPS
n_txns = np.random.poisson(AVG_GLOBAL_TXPS*time_between_blocks)
my_payments = np.random.poisson(MY_PAYMENT_REQUESTS_PS*time_between_blocks)


# Select a fee distribution. The actual sizes don't matter that much, as they
# are relative to the other values, but they should be:
# 1) Non-negative
# 2) Of the correct "shape" you want to model
#
# It's possible to modify the code to model different global v.s. local feerate
# distributions.
#
# Two distributions are provided for comparison -- lognormal, uniform, and
# abscauchy
#
# lognormal intuitively seems close to a realistic model. A bulk of relatively
# low priority transactions and an infinite tail of higher priority stuff.
#
# uniform is "easy" to understand, but less realistic seeming as there should
# fundamentally be less stuff that is "high priority" than low priority and
# the max high priority is uncapped.
#
# abscauchy is used to check that something with a really fat tail of results
# doesn't mess up the methods -- we can't use logcauchy as that would make the
# tail "too fat"
FEE_DISTRIBUTION_TYPE = "lognormal"
MY_FEE_DISTRIBUTION = None
GLOBAL_FEE_DISTRIBUTION = None
if FEE_DISTRIBUTION_TYPE == "uniform":
    MY_FEE_DISTRIBUTION = lambda N: np.random.uniform(0.02, 1, N)
    GLOBAL_FEE_DISTRIBUTION = lambda N: np.random.uniform(0.02, 1, N)
elif FEE_DISTRIBUTION_TYPE == "lognormal":
    MY_FEE_DISTRIBUTION = lambda N: np.random.lognormal(0, 1, N)
    GLOBAL_FEE_DISTRIBUTION = lambda N: np.random.lognormal(0, 1, N)
elif FEE_DISTRIBUTION_TYPE == "abscauchy":
    MY_FEE_DISTRIBUTION = lambda N: np.abs(np.random.standard_cauchy(N))
    GLOBAL_FEE_DISTRIBUTION = lambda N: np.abs(np.random.standard_cauchy(N))
# Pick a priority for each payment from our distributions...
tx_priority = map(GLOBAL_FEE_DISTRIBUTION, n_txns)
# sorted for later performance reasons
my_payment_priority = map(np.sort, map(MY_FEE_DISTRIBUTION, my_payments))



# We also need to apply a secondary distribution of weights to just the global
# context. Assume lognormal is OK for now, with a mean of AVG_WEIGHT.
# AVG_WEIGHT should be around 235-250, as a sanity check we can show what
# assumptions that is roughly implying about txn shape.

AVG_N_OUTPUT = 2
AVG_OUTPUT_WEIGHT = 8 + 32 + 1 + 1
AVG_N_INPUT = 1
AVG_WITNESS_WEIGHT = 32 + 64
AVG_INPUT_WEIGHT = 32+4+4+1
# version, locktime, n inputs, n outputs, sPk length, flags
MISC_TX_DATA = 4 + 4 + 1 + 1 + (1+1)+ 1
AVG_WEIGHT = AVG_OUTPUT_WEIGHT*AVG_N_OUTPUT + AVG_WITNESS_WEIGHT*AVG_N_INPUT + AVG_N_INPUT * AVG_INPUT_WEIGHT + MISC_TX_DATA

tx_weight = [np.random.lognormal(math.log(AVG_WEIGHT/math.sqrt(math.e)), 1, n_txn) for n_txn in n_txns]


# The MemPoolEmulator class accepts up to 2*SIZE_LIMIT transactions, and then
# drops the bottom half when this limit is broached. This roughly approximates
# a mempool with SIZE_LIMIT of SIZE_LIMIT when SIZE_LIMIT >> 1 block. The
# limitation of 2*SIZE_LIMIT is set to balance performance and memory concerns.
from heapq import heappush as push
from heapq import heappushpop as pushpop
from heapq import heappop as pop
from heapq import heapify
class MemPoolEmulator():
    LIMITED_MEMPOOL = True
    SIZE_LIMIT = (MAX_BLOCKWEIGHT/AVG_WEIGHT)*100
    def __init__(self):
        self.mempool = []
    def _trim(self):
        # As an approximation, whenever we are 2x to big, only keep the top.
        if MemPoolEmulator.LIMITED_MEMPOOL and len(self.mempool) > 2 * MemPoolEmulator.SIZE_LIMIT:
            mp_copy = self.mempool
            self.mempool = list(pop(self.mempool) for _ in xrange(MemPoolEmulator.SIZE_LIMIT))
            # Always reinsert our own stuff into our mempool!
            # TODO: Log Rebroadcasts?
            for x in mp_copy:
                if x[1][2]:
                    push(self.mempool, x)
            heapify(self.mempool)
    def add_to_mempool(self, item):
        # include a separate key here so that we don't ever
        # risk returning an inverted priority
        # We invert the priority because the heapq takes the smallest value.
        push(self.mempool, (-item[0], item))
        self._trim()
    def has_txns(self):
        return self.mempool != []
    def pop(self):
        return pop(self.mempool)[1]
    def peek(self):
        return self.mempool[0][1]
    def __len__(self):
        return len(self.mempool)




STRATEGIES = ["BATCH_GROUP_MAX",
              "BATCH_GROUP_MAX_CTV",
              "BATCH_MAX",
              "NOBATCH_MED",
              "BATCH_GROUP_MAX_CTV_MIN_FOLLOWUP",
              "BATCH_GROUP_MAX_DOUBLE_CTV_MIN_FOLLOWUP",
              "BATCH_GROUP_MAX_DOUBLE_CTV_CPFP_FOLLOWUP"]

fig = plt.figure()
n_strats = len(STRATEGIES)
g = (7, n_strats)
plot_offered_fees = plt.subplot2grid(g, (0,0), colspan=n_strats)
plot_accepted_fees = plt.subplot2grid(g, (1,0), colspan=n_strats)
plot_confirmed_payments = plt.subplot2grid(g, (2,0), colspan=n_strats)
plot_fee_bands = [0]*n_strats
for (i,strat) in enumerate(STRATEGIES):
    plot_fee_bands[i] = plt.subplot2grid(g, (3,i), colspan=1)
    plot_fee_bands[i].set_yscale('log')
plot_overpaid = plt.subplot2grid(g, (4,0), colspan=n_strats)
plot_weight_issued = plt.subplot2grid(g, (5,0), colspan=n_strats)
plot_weight_mined = plt.subplot2grid(g, (6,0), colspan=n_strats)

plot_offered_fees.set_title("Offered Fees")
plot_accepted_fees.set_title("Accepted Fees / Offered Fees")
plot_confirmed_payments.set_title("Unconfirmed Payments")
plot_overpaid.set_title("Cumulative Amount Overpaid versus Min Fee in Block Mined")
plot_weight_issued.set_title("Cumulative Weight of Issued Transactions")
plot_weight_mined.set_title("Cumulative Weight of Mined Transactions")

plot_offered_fees.set_yscale('log')
plot_overpaid.set_yscale('log')
plot_weight_issued.set_yscale('log')
plot_weight_mined.set_yscale('log')
for (strategy_idx, strategy) in enumerate(STRATEGIES):
    print "Simulating strategy ", strategy
    mempool = MemPoolEmulator()
    # Tracks the number of my payments that are not yet turned into UTXOS
    my_waiting_payments = 0
    my_waiting_history = np.zeros(SIM_BLOCKS)
    # Tracks the amount of my fees confirmed in a block
    my_confirmed_fees = 0
    my_confirmed_fees_history = np.zeros(SIM_BLOCKS)
    # Tracks the amount of fees spent in the mempool
    my_issued_fee = 0
    my_issued_fee_history = np.zeros(SIM_BLOCKS)
    # Tracks the max/min/median fees for tranasactions in a block
    min_priority = np.zeros(SIM_BLOCKS)
    max_priority = np.zeros(SIM_BLOCKS)
    median_priority = np.zeros(SIM_BLOCKS)
    # Tracks the overpayment (above min feerate) this block
    my_overpaid_this_block = np.zeros(SIM_BLOCKS)
    # Tracks the total weight
    my_weight_mined_this_block = np.zeros(SIM_BLOCKS)
    my_issued_weight_this_block = np.zeros(SIM_BLOCKS)

    for i in xrange(SIM_BLOCKS):
        # invert for priority queue so that the max is the min
        for p in zip(tx_priority[i], tx_weight[i], [False]*len(tx_weight[i]), [1]*len(tx_weight[i])):
            mempool.add_to_mempool(p)

        # Wait until we've mined a single block so we can get a median fee
        # estimate. We could also just peek at the mempool at this point to
        # see what the median is of what *will be mined*, but we can get
        # roughly equivalent by just adding one more SIM_BLOCKS
        if i > 0 and my_payments[i] > 0:
            median = median_priority[i-1]
            my_waiting_payments += my_payments[i]
            # select a strategy for my payments
            if strategy == "NOBATCH_MED":
                # In this strategy we just issue transactions per payment, no
                # batching or anything else fancy. We pay the median fee of the
                # last block.
                weight = AVG_WEIGHT
                for p in xrange(my_payments[i]):
                    # Pay the min of the median or my priority
                    priority = min(median, my_payment_priority[i][p])
                    mempool.add_to_mempool((priority, weight, True, 1))
                    my_issued_fee += weight*priority
                    my_issued_weight_this_block[i] += weight
            if strategy == "BATCH_MAX":
                # In this strategy we batch transactions all together and then
                # we pay the min of the median or max feerate in batch
                hi_priority = max(my_payment_priority[i])
                priority = min(median, hi_priority)
                weight = AVG_WEIGHT - 1*AVG_OUTPUT_WEIGHT + my_payments[i]*AVG_OUTPUT_WEIGHT
                mempool.add_to_mempool((priority, weight, True, my_payments[i]))
                my_issued_fee += weight*priority
                my_issued_weight_this_block[i] += weight
            if strategy == "BATCH_GROUP_MAX":
                # In this strategy we batch transactions of similar feerate
                # together and then we pay the min of the median or max feerate
                # in each batch
                bins, bin_edges = np.histogram(my_payment_priority[i], bins="auto", range=(0, median))
                bins[-1] += my_payments[i] - sum(bins)
                assert sum(bins) == my_payments[i]
                for (bin_idx, count) in enumerate(bins):
                    if count == 0: continue
                    lo_priority, hi_priority = bin_edges[bin_idx:bin_idx+2]
                    # The hi_priority is one too high, find the predecessor
                    # unless it's the last bin
                    if bin_idx != len(bins) -1:
                        hi_priority = my_payment_priority[i][np.searchsorted(my_payment_priority[i], hi_priority) - 1]
                    # The hi_priority is one too high, find the predecessor
                    priority = hi_priority
                    # Pay the min of the median or max priority in batch
                    priority = min(median, priority)
                    weight = AVG_WEIGHT - 1*AVG_OUTPUT_WEIGHT + count*AVG_OUTPUT_WEIGHT
                    mempool.add_to_mempool((priority, weight, True, count))
                    my_issued_fee += weight*priority
                    my_issued_weight_this_block[i] += weight
            if strategy == "BATCH_GROUP_MAX_CTV":
                # In this strategy we batch transactions of similar feerate
                # together and then we pay the min of the median or max feerate
                # in each batch
                # But instead of making independent batches, we put them through
                # one layer of a high fee paying CTV root.

                bins, bin_edges = np.histogram(my_payment_priority[i], bins="auto", range=(0, median))
                bins[-1] += my_payments[i] - sum(bins)
                assert sum(bins) == my_payments[i]

                # Pay High Fee for Root
                priority = min(median, max(my_payment_priority[i]))
                n_bins = sum(b != 0 for b in bins)
                root_weight = AVG_WEIGHT - 1*AVG_OUTPUT_WEIGHT + AVG_OUTPUT_WEIGHT*n_bins
                # counts for everyone being paid when confirmed
                mempool.add_to_mempool((priority, root_weight, True, my_payments[i]))
                my_issued_fee += root_weight*priority
                my_issued_weight_this_block[i] += root_weight

                # Pay High Fee for each bin
                for (bin_idx, count) in enumerate(bins):
                    # don't make a txn for empty buckets
                    if count == 0: continue
                    # If the bucket is a single element, just directly include
                    # it in the root & skip a CTV extra step
                    if count == 1: continue
                    lo_priority, hi_priority = bin_edges[bin_idx:bin_idx+2]
                    # The hi_priority is one too high, find the predecessor
                    # unless it's the last bin
                    if bin_idx != len(bins) -1:
                        hi_priority = my_payment_priority[i][np.searchsorted(my_payment_priority[i], hi_priority) - 1]
                    priority = hi_priority
                    # Pay the min of the median or max priority in batch
                    priority = min(median, priority)
                    weight = AVG_WEIGHT - AVG_N_OUTPUT*AVG_OUTPUT_WEIGHT - AVG_WITNESS_WEIGHT*AVG_N_INPUT + count*AVG_OUTPUT_WEIGHT
                    # It's mine, but no one got paid because we counted the root
                    mempool.add_to_mempool((priority, weight, True, 0))
                    my_issued_fee += weight*priority
                    my_issued_weight_this_block[i] += weight
            if strategy == "BATCH_GROUP_MAX_CTV_MIN_FOLLOWUP":
                # Pay 10-block low (but mineable) fee for each bin
                min_fee = min(min_priority[max(i-10,0):i])
                # In this strategy we batch transactions of similar feerate
                # together and then we pay the min of the min feerate over the
                # last ten blocks or max feerate in each batch
                # But instead of making independent batches, we put them through
                # one layer of a high fee paying CTV root with n_bins outputs.
                #
                # We cut through bins with only one element
                bins, bin_edges = np.histogram(my_payment_priority[i],
                                               bins="auto", range=(0, min_fee))
                bins[-1] += my_payments[i] - sum(bins)
                assert sum(bins) == my_payments[i]

                # Pay High Fee for Root
                priority = min(median, max(my_payment_priority[i]))
                n_bins = sum(b != 0 for b in bins)
                root_weight = AVG_WEIGHT - 1*AVG_OUTPUT_WEIGHT + AVG_OUTPUT_WEIGHT*n_bins
                # counts for everyone being paid when confirmed
                mempool.add_to_mempool((priority, root_weight, True, my_payments[i]))
                my_issued_fee += root_weight*priority
                my_issued_weight_this_block[i] += root_weight

                for (bin_idx, count) in enumerate(bins):
                    # don't make a txn for empty buckets
                    if count == 0: continue
                    # If the bucket is a single element, just directly include
                    # it in the root & skip a CTV extra step
                    if count == 1: continue
                    lo_priority, hi_priority = bin_edges[bin_idx:bin_idx+2]
                    priority = hi_priority
                    # Pay the min of the median or max priority in batch
                    priority = min(min_fee, priority)
                    weight = AVG_WEIGHT - AVG_N_OUTPUT*AVG_OUTPUT_WEIGHT - AVG_WITNESS_WEIGHT*AVG_N_INPUT + count*AVG_OUTPUT_WEIGHT
                    # It's mine, but no one got paid because we counted the root
                    mempool.add_to_mempool((priority, weight, True, 0))
                    my_issued_fee += weight*priority
                    my_issued_weight_this_block[i] += weight
            if strategy == "BATCH_GROUP_MAX_DOUBLE_CTV_MIN_FOLLOWUP":
                # In this strategy we batch transactions of similar feerate
                # together and then we pay the min of the min feerate over the
                # last ten blocks or max feerate in each batch
                # But instead of making independent batches, we put them through
                # two layers of CTV: a high fee paying CTV root with one output,
                # and then a low paying CTV root with n_bins outputs.
                #
                # We cut through bins with only one element

                # Pay 10-block low (but mineable) fee for each bin and second
                # root
                min_fee = min(min_priority[max(i-10,0):i])

                bins, bin_edges = np.histogram(my_payment_priority[i],
                                               bins="auto", range=(0, min_fee))
                bins[-1] += my_payments[i] - sum(bins)
                assert sum(bins) == my_payments[i]

                # Pay High Fee for first Root
                priority = min(median, max(my_payment_priority[i]))
                n_bins = sum(b != 0 for b in bins)
                root_weight = AVG_WEIGHT - 1*AVG_OUTPUT_WEIGHT
                # counts for everyone being paid when confirmed
                mempool.add_to_mempool((priority, root_weight, True, my_payments[i]))
                my_issued_fee += root_weight*priority
                my_issued_weight_this_block[i] += root_weight

                # Second root with low fee
                second_root_weight = AVG_WEIGHT - 1*AVG_OUTPUT_WEIGHT + AVG_OUTPUT_WEIGHT*n_bins - AVG_WITNESS_WEIGHT*AVG_N_INPUT
                priority = min_fee
                mempool.add_to_mempool((priority, second_root_weight, True, 0))
                my_issued_fee += second_root_weight*priority
                my_issued_weight_this_block[i] += second_root_weight

                for (bin_idx, count) in enumerate(bins):
                    # don't make a txn for empty buckets
                    if count == 0: continue
                    # If the bucket is a single element, just directly include
                    # it in the root & skip a CTV extra step
                    if count == 1: continue
                    lo_priority, hi_priority = bin_edges[bin_idx:bin_idx+2]
                    priority = hi_priority
                    # Pay the min of the median or max priority in batch
                    priority = min(min_fee, priority)
                    weight = AVG_WEIGHT - AVG_N_OUTPUT*AVG_OUTPUT_WEIGHT - AVG_WITNESS_WEIGHT*AVG_N_INPUT + count*AVG_OUTPUT_WEIGHT
                    # It's mine, but no one got paid because we counted the root
                    mempool.add_to_mempool((priority, weight, True, 0))
                    my_issued_fee += weight*priority
                    my_issued_weight_this_block[i] += weight

            if strategy == "BATCH_GROUP_MAX_DOUBLE_CTV_CPFP_FOLLOWUP":
                # In this strategy we batch transactions of similar feerate
                # together and then we pay 0 (leaving it up to CPFP)
                # But instead of making independent batches, we put them through
                # two layers of CTV: a high fee paying CTV root with one output,
                # and then a 0 paying CTV root with n_bins outputs.
                #
                # We cut through bins with only one element
                #
                # This puts people into priority bins, but allows CPFP to commit
                # to the fees later avoiding over-offering of fees.

                # Pay 10-block low (but mineable) fee for each bin and second
                # root
                min_fee = min(min_priority[max(i-10,0):i])

                bins, bin_edges = np.histogram(my_payment_priority[i],
                                               bins="auto", range=(0, min_fee))
                bins[-1] += my_payments[i] - sum(bins)
                assert sum(bins) == my_payments[i]

                # Pay High Fee for first Root
                priority = min(median, max(my_payment_priority[i]))
                n_bins = sum(b != 0 for b in bins)
                root_weight = AVG_WEIGHT - 1*AVG_OUTPUT_WEIGHT
                # counts for everyone being paid when confirmed
                mempool.add_to_mempool((priority, root_weight, True, my_payments[i]))
                my_issued_fee += root_weight*priority
                my_issued_weight_this_block[i] += root_weight

                # Second root with low fee
                second_root_weight = AVG_WEIGHT - 1*AVG_OUTPUT_WEIGHT + AVG_OUTPUT_WEIGHT*n_bins - AVG_WITNESS_WEIGHT*AVG_N_INPUT
                priority = 0
                mempool.add_to_mempool((priority, second_root_weight, True, 0))
                my_issued_fee += second_root_weight*priority
                my_issued_weight_this_block[i] += second_root_weight

                for (bin_idx, count) in enumerate(bins):
                    # don't make a txn for empty buckets
                    if count == 0: continue
                    # If the bucket is a single element, just directly include
                    # it in the root & skip a CTV extra step
                    if count == 1: continue
                    priority = 0
                    weight = AVG_WEIGHT - AVG_N_OUTPUT*AVG_OUTPUT_WEIGHT - AVG_WITNESS_WEIGHT*AVG_N_INPUT + count*AVG_OUTPUT_WEIGHT
                    # It's mine, but no one got paid because we counted the root
                    mempool.add_to_mempool((priority, weight, True, 0))
                    my_issued_fee += weight*priority
                    my_issued_weight_this_block[i] += weight


        blockweight = 0
        postponed = []
        accepted = []
        my_accepted_priority = []
        my_accepted_weight = []
        while blockweight < MAX_BLOCKWEIGHT and mempool.has_txns():
            (priority, w, is_mine, n_paid) = mempool.peek()
            if w + blockweight < MAX_BLOCKWEIGHT:
                mempool.pop()
                blockweight += w
                accepted.append(priority)
                if is_mine:
                    my_waiting_payments -= n_paid
                    my_confirmed_fees += priority*w
                    my_accepted_priority.append(priority)
                    my_accepted_weight.append(w)
                    my_weight_mined_this_block[i] += w
            elif MAX_BLOCKWEIGHT - blockweight < 400:
                break
            elif len(postponed) > 400:
                break
            elif len(mempool) == 0:
                break
            else:
                postponed.append(mempool.pop())
        for p in postponed:
            mempool.add_to_mempool(p)
        min_priority[i] = min(accepted)
        my_overpaid_this_block[i] = np.dot((np.array(my_accepted_priority) -
                                  min_priority[i]),
                                        np.array(my_accepted_weight))
        max_priority[i] = max(accepted)
        median_priority[i] = np.median(accepted)
        my_waiting_history[i] = my_waiting_payments
        my_confirmed_fees_history[i] = my_confirmed_fees
        my_issued_fee_history[i] = my_issued_fee
    SECONDS_TO_DAYS = 60*60*24
    X = absolute_block_arrivals/SECONDS_TO_DAYS
    plot_accepted_fees.plot(X,
                            my_confirmed_fees_history/my_issued_fee_history, label =strategy)
    plot_offered_fees.plot(X, my_issued_fee_history, label= strategy)
    plot_confirmed_payments.plot(X, my_waiting_history,
                                 label=strategy)
    plot_fee_bands[strategy_idx].plot(X, min_priority, label="min")
    plot_fee_bands[strategy_idx].plot(X, median_priority, label="med")
    plot_fee_bands[strategy_idx].plot(X, max_priority, label="max")
    plot_overpaid.plot(X, np.cumsum(my_overpaid_this_block), label=strategy)
    plot_weight_issued.plot(X, np.cumsum(my_issued_weight_this_block),
                            label=strategy)
    plot_weight_mined.plot(X, np.cumsum(my_weight_mined_this_block),
                            label=strategy)

plot_offered_fees.legend(bbox_to_anchor=(1.05, 1), loc='upper center', borderaxespad=0.)
plot_accepted_fees.legend(bbox_to_anchor=(1.05, 1), loc='upper center', borderaxespad=0.)
plot_confirmed_payments.legend(bbox_to_anchor=(1.05, 1), loc='upper center', borderaxespad=0.)
plot_overpaid.legend(bbox_to_anchor=(1.05, 1), loc='upper center', borderaxespad=0.)
plot_weight_issued.legend(bbox_to_anchor=(1.05, 1), loc='upper center', borderaxespad=0.)
plot_weight_mined.legend(bbox_to_anchor=(1.05, 1), loc='upper center', borderaxespad=0.)
for (i, s) in enumerate(STRATEGIES):
    plot_fee_bands[i].set_title("Fee Bands "+s)
    plot_fee_bands[i].legend(loc='upper center', borderaxespad=0.)
plt.show()



