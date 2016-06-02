# The MIT License (MIT)
#
# Copyright (c) 2016, Jack Liu
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import time, datetime, os
from collections import OrderedDict
import csv, json, re, sys
import requests
import random
import operator
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo_ochl
from matplotlib.dates import HourLocator, MinuteLocator, SecondLocator, DateFormatter
from matplotlib.patches import Ellipse

ctrl = {}

class Quote(object):
    def dump(self):
        print self.dt, self.o, self.h, self.l, self.c, self.v
        
    def __init__(self, dt, o, h, l, c, v):
        assert(h >= l)
        self.dt = dt  # time stamp
        self.o = o;   # open privde
        self.h = h;   # high
        self.l = l;   # low
        self.c = c;   # close
        self.v = v;   # volume

    def get_day(self):
        return self.dt.day

    def get_median(self):
        return (self.o + self.c) / 2

    # Return the median price of this quote wrt ref price.
    def get_ratio(self,ref_score):
        return  ((self.get_median() - ref_score) / ref_score)

    def get_normalized_dt(self):
        return datetime.datetime(1971, 1, 1, \
                                 self.dt.hour, self.dt.minute, self.dt.second) 

#
# Class Plot
#
class Plot(object):
    def __init__(self, chart_type):
        self.markers = ['o', 'v', '^', 's', 'p', '*', 'h', 'H', 'D', 'd']
        self.ls = ['dashed', 'dashdot', 'dotted']
        self.hours   = HourLocator()    # every hour
        self.minutes = MinuteLocator()  # every minute
        self.seconds = SecondLocator()  # every second
        self.hoursFmt = DateFormatter('%H')
        self.fig, self.ax = plt.subplots(figsize=(20, 10))
        self.chart_type = chart_type
        return

    def plot_scores(self, dates, scores, mfc, marker, quote):
        self.ax.plot_date(dates, scores,
                          ls=random.choice(self.ls), marker=marker,
                          markersize=5.0, markerfacecolor=mfc,
                          label=str(quote.dt.month)+'/'+str(quote.dt.day))

        self.ax.text(quote.dt, quote.c, str(quote.c), fontsize=12, color='g')
        return

    def plot_buys(self, buy_quotes, quotes_2_scores):
        if not buy_quotes:
            return
        norm_dates  = [q.get_normalized_dt() for q in buy_quotes]
        scores = [score for quote, score in quotes_2_scores.iteritems() \
                  for buy_quote in buy_quotes if quote.dt == buy_quote.dt]
        size = [200.0 for _ in buy_quotes]
        self.ax.scatter(norm_dates, scores, s=size, color='b', alpha=0.8)
        return

    def plot_future(self, future_scores):
        (dates, scores) = zip(*future_scores)

        self.ax.plot_date(dates, scores,
                          ls=random.choice(self.ls), marker='D',
                          markersize=5.0, markerfacecolor='g',
                          label='future')
        return

    def format_ticks(self):
        self.ax.xaxis.set_major_locator(self.hours)
        self.ax.xaxis.set_major_formatter(self.hoursFmt)
        self.ax.xaxis.set_minor_locator(self.minutes)
        self.ax.autoscale_view()

        # format the coords message box
        def price(x):
            return '$%.3f' % x
        self.ax.fmt_xdata = DateFormatter('%H-%M-%S')
        self.ax.fmt_ydata = price
        self.ax.grid(True)

        self.fig.autofmt_xdate()

        plt.legend(loc='best', shadow=True)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
        
        return

    def format_labels(self, interval_seconds):
        plt.ylabel(self.chart_type)
        plt.xlabel('Interval ' + str(interval_seconds / 60.0) + ' min')
        return

    def format_title(self, title):
        plt.title(title)
        return

    def show(self):
        sys.stdout.flush()
        plt.show()
        return

    def annotate(self, x, y, xytext):
        self.ax.annotate('%.3f' % y, xy=(x, y), xycoords='data',
                         bbox=dict(boxstyle="round4", fc="w", alpha=0.75),
                         xytext=xytext, textcoords='offset points', size=14,
                         arrowprops=dict(arrowstyle="fancy",
                                         fc="0.3", ec="none",
                                         patchB=Ellipse((2, -1), 0.5, 0.5),
                                         connectionstyle="angle3,angleA=0,angleB=-90"),
        )
        return
#
#
#
class Stock(object):
    def __repr__(self):
        return "Stock()"
    
    def __str__(self):
        return self.symbol
        
    def __init__(self, symbol, interval_seconds, buys, sells):
        self.symbol = symbol
        # The page_num, or key, of self.book is the yr_mo_day, and the
        # content of each page is a list of quotes on that day.
        self.book = OrderedDict()
        self.interval_seconds = interval_seconds
        self.knn_candidate_set = set()

        # Create buys and sells records, and convert them
        # from string to datetime type
        self.buys = []
        self.sells = []
        
        for buy in buys:
            self.buys.append(datetime.datetime.strptime(buy, "%Y-%m-%d %H:%M:%S"))

        for sell in sells:
            self.sells.append(datetime.datetime.strptime(sell, "%Y-%m-%d %H:%M:%S"))

        self.buys.sort()
        self.sells.sort()

        return

    # Given a list of quotes, return a sub-list of it that have buy events
    def get_buy_quotes(self, quotes):
        assert(quotes)
        results = []

        for buy in self.buys:
            if (buy > quotes[-1].dt) or (buy < quotes[0].dt):
                continue
            for quote in quotes:
                if quote.dt >= buy:
                    results.append(quote)
                    break;

        return results;

    # Given a quote, return the page_num, or key, where this qutoe should
    # belong to.
    def get_page_num_str(self, quote):
        return str(quote.dt.year) + '_' + str(quote.dt.month) + '_' + str(quote.dt.day)

    # Append a new quote according to its day.
    def append(self, quote):
        # page_num is the key of self.book
        page_num = self.get_page_num_str(quote)
        if not self.book.has_key(page_num):
            self.book[page_num] = []
        self.book[page_num].append(quote)
        return

    def dump(self):
        print self.symbol, len(self.quotes)
        for quote in self.quotes:
            quote.dump()

    def __repr__(self):
        return self.to_csv()

    #
    # Compute KNN candidates that meet today's criteria.
    #
    def prepare_knn_candidate_set(self):
        if len(self.book) < 2:
            return 0

        open_price = self.get_today_open_price();
        prev_close_price = self.get_yesterday_close_price();
        ref_ratio = (open_price - prev_close_price) / prev_close_price
        print "KNN ratio: ", prev_close_price, open_price, ref_ratio

        prev_close_price = 1
        
        for page_num, quotes in self.book.iteritems():
            open_price = quotes[0].o
            ratio = (open_price - prev_close_price) / prev_close_price

            if (abs(ref_ratio) <= 0.04) and (abs(ratio) <= 0.04):
                # in [-0.04, 0.04] range
                self.knn_candidate_set.add(quotes[0].dt);
            elif (ref_ratio > 0.04) and (ratio > 0.04):
                # in [0.04, +] range
                self.knn_candidate_set.add(quotes[0].dt);
            elif (ref_ratio < 0.04) and (ratio < 0.04):
                # in [-, -0.04] range
                self.knn_candidate_set.add(quotes[0].dt);
            else:
                self.knn_candidate_set.add(quotes[0].dt);

            prev_close_price = quotes[-1].c
            
        return len(self.knn_candidate_set)
    
    #
    # Should we consider these quotes a valid candidate for KNN
    #
    def is_knn_candidate(self, quotes):
        return quotes[0].dt in self.knn_candidate_set;
    
    #
    # Given a sequence of quotes, retun a list of
    # scores depending on ChartType.
    #
    def compute_display_scores(self, quotes, chart_type):
        # Use close price as score
        if chart_type == 'close':
            return [q.c for q in quotes]

        # Use median price
        if chart_type == 'median':
            return [q.get_median() for q in quotes]
        
        # K-nearest neighbors
        if chart_type == 'knn':
            if self.is_knn_candidate(quotes):
                return [q.get_ratio(quotes[0].o) for q in quotes]
            else:
                return []

        # Default will use close price as score
        return [q.c for q in quotes]

    def get_first_page(self):
        return self.book.itervalues().next()

    def get_latest_quote(self):
        page_num = self.book.keys()[-1];
        return self.book[page_num][-1]
    
    def get_today_open_price(self):
        assert(len(self.book) >= 1)
        page_num = self.book.keys()[-1];        
        return self.book[page_num][0].o

    def get_yesterday_close_price(self):
        assert(len(self.book) >= 2)
        page_num = self.book.keys()[-2];        
        return self.book[page_num][-1].c

    def is_last_page(self, page_num):
        return page_num == self.book.keys()[-1];

    def get_local_time(self):
        now = datetime.datetime.now()
        local_hr = now.hour + ctrl['UTC']
        local_day = now.day
        if local_hr < 0:
            local_hr += 24
            local_day -= 1
        return (local_day, local_hr, now.minute, now.second)

    def get_num_days_ago(self, quote):
        return (datetime.datetime.now() - quote.dt).days

    def is_market_closed(self):
        (local_day, local_hr, local_min, local_sec) = self.get_local_time()
        last_quote_dt = self.get_latest_quote().dt
        if local_day != last_quote_dt.day:
            return True
        if local_hr < 6 or local_hr >= 13:
            return True
        return False

    def is_market_open(self):
        return not self.is_market_closed()

    def write2csv(self):
        if not self.book:
            return
        
        fname = self.symbol + ".csv"
        print "Create", fname

        with open(fname, 'wb') as f:
            first_page = self.get_first_page();
            keys = first_page[0].__dict__.keys()
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()

            last_quote_dt = datetime.datetime.fromtimestamp(0);
            for page_num, quotes in self.book.iteritems():
                # Make sure quotes are listed in order
                assert(last_quote_dt < quotes[0].dt)
                last_quote_dt = quotes[0].dt
                for quote in quotes:
                    w.writerow(quote.__dict__)
        return

    #
    # Return reference datetime which shows where the quote is at this moment
    #
    def get_ref_datetime(self) :
        (_, local_hr, local_min, local_sec) = self.get_local_time()
        
        if self.is_market_open():
            ref_datetime = datetime.datetime(1971, 1, 1, \
                                             local_hr, local_min, local_sec)
        else:
            ref_datetime = datetime.datetime(1971, 1, 1, 6, 20)
        return ref_datetime

    #
    # Compute today's future scores based on historical scores, volume, and
    # today's opening score.
    #
    def compute_future_scores(self, hist, today_opening_score, rel):
        future = []
        hist_opening_scores = hist.itervalues().next()
        future_score = today_opening_score

        def geomean(nums):
            return reduce(lambda x, y: x*y, nums)**(1.0/len(nums))
    
        for dt, scores_and_volume in hist.iteritems():
            # An effective_record is a tuple of (hist_score, volume, hist_opening_score)
            effective_records = [(a[0], float(a[1]), b[0]) for (a, b) in \
                                 zip(scores_and_volume, hist_opening_scores) if rel(a[0], b[0])]
            total_volume = sum([rec[1] for rec in effective_records])
            
            if today_opening_score < 1.0:
                # No need to compute ratio again if chart type is knn
                delta_score = sum(([(rec[0] - rec[2]) * (rec[1] / total_volume) for rec in effective_records]))
                future_score = today_opening_score + delta_score
            else:
                # Use ratio wrt hist_opening
                ratio = sum(([(rec[0] / rec[2]) * (rec[1] / total_volume) for rec in effective_records]))
                future_score = today_opening_score * ratio
                    
            future.append((dt, future_score))

        return future
    
    # Plot the history for current symbol
    def plot(self, chart_type='close'):
        if not self.book:
            print "Not enough data to plot"
            return

        # Prepare data for KNN
        if chart_type == 'knn':
            if self.prepare_knn_candidate_set() < 2:
                print "No candidate for KNN plot"
                return

        plot = Plot(chart_type);

        num_days = len(self.book)
        gradient = 1.0;

        ref_datetime = self.get_ref_datetime()
        print "Reference time:", ref_datetime

        last_quote_dt = datetime.datetime.fromtimestamp(0);
        opening_score = 0.0
        
        #
        # Walk thru each day
        #

        # historical[normalized_dt] is a list of scores happened at that time.
        # We use historcial later to estimate future scores
        historical = OrderedDict()

        for page_num, quotes in self.book.iteritems():
            scores = self.compute_display_scores(quotes, chart_type)
            if not scores:
                continue;

            # Keep the mapping for later reference.
            quotes_2_scores = OrderedDict(zip(quotes, scores))
            opening_score = scores[0]

            # Update historical
            for quote, score in quotes_2_scores.iteritems():
                _norm_dt = quote.get_normalized_dt()
                if not historical.has_key(_norm_dt):
                    historical[_norm_dt] = []
                historical[_norm_dt].append((score, quote.v))

            # Make sure quotes are listed in ascending order
            assert(last_quote_dt < quotes[0].dt)
            last_quote_dt = quotes[0].dt
            
            # Normalize the year/month/day, since the chart only cares about hr/min/sec
            norm_dates  = [q.get_normalized_dt() for q in quotes]

            # Quotes from last page worths more attention.
            if self.is_last_page(page_num):
                mfc = "red"
                marker = 'D'
            else:
                mfc = str(gradient)
                marker = random.choice(plot.markers)

            # Only show details for the past 7 days
            if self.get_num_days_ago(quotes[0]) > 7:
                continue

            plot.plot_scores(norm_dates, scores, mfc, marker, quotes[0])

            #
            # Highlight the buys using big green dots
            #
            plot.plot_buys(self.get_buy_quotes(quotes), quotes_2_scores)

            # Adjust gradient for next page's quotes
            gradient -= (1.0 / float(num_days - 1));


        # Compute future scores (upper bounds and lower bounds, based on historical data
        future_scores_lb = self.compute_future_scores(historical, opening_score, operator.le)
        future_scores_ub = self.compute_future_scores(historical, opening_score, operator.ge)

        # Plot future scores
        plot.plot_future(future_scores_ub)
        plot.plot_future(future_scores_lb)

        #
        # Print out prediction before showing the chart
        #
        latest_quote = self.get_latest_quote();
        print self.symbol, "now @", latest_quote.c
        
        # format the ticks and labels
        plot.format_ticks()
        plot.format_labels(self.interval_seconds)
        title = self.symbol + " in last " + str(num_days) + " days" + " @" + str(latest_quote.c)
        plot.format_title(title)

        #
        # Annotate the min/max points
        #
        (x1, y1) = max(future_scores_ub, key=operator.itemgetter(1))
        (x2, y2) = min(future_scores_lb, key=operator.itemgetter(1))
        plot.annotate(x1, y1, (-80, 60))
        plot.annotate(x2, y2, (-80, -60))
        
        # Flush whatever we have, and draw it!
        plot.show()

        return

#
# Collect intraday quote for a symbol.
#
def CollectIntradayQuote(record, interval_seconds, num_days):
    symbol = record["Symbol"]
    stock = Stock(symbol, interval_seconds, buys=record.get("Buy", []),
                  sells=record.get("Sell", []))
    url = ctrl['URL']
    url += "q={0}&i={1}&p={2}d&f=d,o,h,l,c,v".format(symbol,interval_seconds,num_days)
    print "Query", url
    csv = requests.get(url).text.encode('utf-8').split('\n')

    _, timezone_offset = csv[6].split('=')
    # Adjust timezone wrt UTC
    timezone_offset = (0 * float(timezone_offset)) + (ctrl["UTC"] * 60 * 60)

    for row in csv[7:]:
        fields = row.split(',')
        if len(fields) != 6:
            continue;
        
        # COLUMNS=DATE,CLOSE,HIGH,LOW,OPEN,VOLUME
        offset = fields[0]
        if offset.startswith('a'):
            day = int(offset[1:])
            offset = 0
        else:
            offset = int(offset)

        dt = datetime.datetime.fromtimestamp(day+(interval_seconds*offset)+timezone_offset)

        # Create a new quote
        quote = Quote(dt, float(fields[4]), float(fields[2]), float(fields[3]), \
                      float(fields[1]), int(fields[5]))

        # Append this new quote to current stock
        stock.append(quote)

    stock.write2csv()
    return stock

#
# main()
#

try:
    with open('stock.json') as f:
        ctrl = json.load(f);
except IOError as e:
    sys.exit( "I/O error({0}): {1}".format(e.errno, e.strerror) + ": stock.json")

for record in ctrl["Records"]:
    stock = CollectIntradayQuote(record, ctrl["Interval"], ctrl["Days"])
    stock.plot(record['ChartType'].lower())
