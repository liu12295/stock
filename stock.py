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
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.finance import quotes_historical_yahoo_ochl
from matplotlib.dates import HourLocator, MinuteLocator, SecondLocator, DateFormatter

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
        return np.median(np.array([self.o,self.h,self.l,self.c]))

    # Return the median price of this quote wrt ref price.
    def get_ratio(self,ref):
        return  ((self.get_median() - ref) / ref)

    def get_normalized_dt(self):
        return datetime.datetime(1971, 1, 1, \
                                 self.dt.hour,self.dt.minute, second=self.dt.second) 

class Stock(object):
    def __repr__(self):
        return "Stock()"
    
    def __str__(self):
        return self.symbol
        
    def __init__(self, symbol, interval_seconds):
        self.symbol = symbol
        # The page_num, or key, of self.book is the yr_mo_day, and the
        # content of each page is a list of quotes on that day.
        self.book = OrderedDict()
        self.interval_seconds = interval_seconds
        self.knn_candidate_set = set()

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
        sys.stdout.flush()
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
    def compute_scores(self, quotes):
        # Use close price as score
        if ctrl['ChartType'].lower() == "close":
            return [q.c for q in quotes]

        # Use median price
        if ctrl['ChartType'].lower() == "median":
            return [q.get_median() for q in quotes]
        
        # K-nearest neighbors
        if ctrl['ChartType'].lower() == "knn":
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

    # Plot the history for current symbol
    def plot(self):
        if not self.book:
            print "Not enough data to plot"
            return

        # Prepare data for KNN
        if ctrl['ChartType'].lower() == "knn":
            if self.prepare_knn_candidate_set() < 2:
                print "No candidate for KNN plot"
                return

        markers = ['o', 'v', '^', 's', 'p', '*', 'h', 'H', 'D', 'd']
        ls = ['dashed', 'dashdot', 'dotted']
    
        hours   = HourLocator()    # every hour
        minutes = MinuteLocator()  # every minute
        seconds = SecondLocator()  # every second
        hoursFmt = DateFormatter('%H')

        fig, ax = plt.subplots(figsize=(20, 10))

        num_days = len(self.book)
        gradient = 1.0;

        #
        # Walk thru each day
        #
        last_quote_dt = datetime.datetime.fromtimestamp(0);

        for page_num, quotes in self.book.iteritems():
            scores = self.compute_scores(quotes)

            if not scores:
                continue;

            # Make sure quotes are listed in order
            if last_quote_dt:
                assert(last_quote_dt < quotes[0].dt)
                last_quote_dt = quotes[0].dt
            
            # Normalize the year/month/day, since the chart only cares about hr/min/sec
            dates  = [q.get_normalized_dt() for q in quotes]
            last_quote = quotes[0]

            # Quotes from last page worths more attention.
            if self.is_last_page(page_num):
                mfc = "red"
                marker = 'D'
            else:
                mfc = str(gradient)
                marker = random.choice(markers)

            ax.plot_date(dates, scores, \
                         ls=random.choice(ls), marker=marker, \
                         markerfacecolor=mfc, \
                         label=str(last_quote.dt.month)+'/'+str(last_quote.dt.day))

            ax.text(last_quote.dt, last_quote.c, str(last_quote.c), fontsize=12, color='g')

            gradient -= (1.0 / float(num_days - 1));

        # format the ticks
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(hoursFmt)
        ax.xaxis.set_minor_locator(minutes)
        ax.autoscale_view()

        # format the coords message box
        def price(x):
            return '$%1.2f' % x
        ax.fmt_xdata = DateFormatter('%H-%M-%S')
        ax.fmt_ydata = price
        ax.grid(True)

        fig.autofmt_xdate()
        
        plt.legend(loc='best', shadow=True)
        plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
        plt.ylabel(ctrl['ChartType'])
        plt.xlabel('Interval ' + str(self.interval_seconds / 60.0) + ' min')

        q = self.get_latest_quote();
        plt.title(self.symbol + " in last " + str(num_days) + " days" + " @" + str(q.c))
        
        plt.show()
        return

#
# Collect intraday quote for a symbol.
#
def CollectIntradayQuote(symbol, interval_seconds, num_days):
    stock = Stock(symbol, interval_seconds)
    url = ctrl['URL']
    url += "q={0}&i={1}&p={2}d&f=d,o,h,l,c,v".format(symbol,interval_seconds,num_days)
    csv = requests.get(url).text.encode('utf-8').split('\n')

    _, timezone_offset = csv[6].split('=')
    # Adjust timezone wrt UTC
    timezone_offset = float(timezone_offset) + (ctrl["UTC"] * 60 * 60)

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

print "Now", datetime.datetime.now()

for symbol in ctrl["Symbols"]:
    stock = CollectIntradayQuote(symbol["symbol"], ctrl["Interval"], ctrl["Days"])
    stock.plot()
