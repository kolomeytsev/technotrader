import calendar
import datetime


class CalendarMOEX:

    def mark_holiday(self, year, month, day):
        self.schedule[year][month][day] = (
            False, 
            datetime.datetime(year=year, month=month, day=day, hour=0, minute=0),
            datetime.datetime(year=year, month=month, day=day, hour=0, minute=0)
        )

    def mark_workday(self, year, month, day):
        self.schedule[year][month][day] = (
            True,
            datetime.datetime(year=year, month=month, day=day, hour=10, minute=0),
            datetime.datetime(year=year, month=month, day=day, hour=18, minute=40)
        )

    def __getitem__(self, dt):
        if isinstance(dt, str):
            dt = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        if isinstance(dt, int):
            dt = datetime.datetime.fromtimestamp(dt)
        return self.schedule[dt.year][dt.month][dt.day][1] <= dt < \
                self.schedule[dt.year][dt.month][dt.day][2]

    def __init__(self):
        self.schedule = dict()

        for year in range(2012, 2020):
            self.schedule[year] = dict()
            for month in range(1, 13):
                self.schedule[year][month] = dict()
                for day in range(1, calendar.monthrange(year, month)[1] + 1):
                    date = datetime.date(year, month, day)
                    wd = date.weekday()
                    if wd > 4:
                        self.mark_holiday(year, month, day)
                    else:
                        self.mark_workday(year, month, day)
            self.mark_holiday(year, 2, 23)
            self.mark_holiday(year, 3, 8)
            self.mark_holiday(year, 5, 1)
            self.mark_holiday(year, 5, 9)
            self.mark_holiday(year, 6, 12)
            self.mark_holiday(year, 11, 4)

        # 2012
        self.mark_holiday(2012, 1, 2)
        self.mark_holiday(2012, 3, 9)
        self.mark_workday(2012, 3, 11)
        self.mark_workday(2012, 4, 28)
        self.mark_holiday(2012, 4, 30)
        self.mark_workday(2012, 5, 5)
        self.mark_workday(2012, 5, 12)
        self.mark_workday(2012, 6, 9)
        self.mark_holiday(2012, 6, 11)
        self.mark_holiday(2012, 11, 5)
        self.mark_holiday(2012, 12, 31)
        # 2013
        for i in range(1, 8):
            self.mark_holiday(2013, 1, i)
        self.mark_holiday(2013, 12, 31)
        # 2014
        for i in range(1, 8):
            self.mark_holiday(2014, 1, i)
        self.mark_workday(2014, 1, 6)
        self.mark_holiday(2014, 3, 10)
        self.mark_holiday(2014, 12, 31)
        # 2015
        for i in range(1, 8):
            self.mark_holiday(2015, 1, i)
        self.mark_workday(2015, 1, 5)
        self.mark_workday(2015, 1, 6)
        self.mark_holiday(2015, 3, 9)
        self.mark_holiday(2015, 5, 4)
        self.mark_holiday(2015, 5, 11)
        self.mark_holiday(2015, 12, 31)
        # 2016
        self.mark_holiday(2016, 1, 1)
        self.mark_holiday(2016, 1, 7)
        self.mark_holiday(2016, 1, 8)
        self.mark_workday(2016, 2, 20)
        self.mark_holiday(2016, 5, 2)
        self.mark_holiday(2016, 5, 3)
        self.mark_holiday(2016, 6, 13)
        # 2017
        self.mark_holiday(2017, 1, 2)
        self.mark_holiday(2017, 5, 8)
        self.mark_holiday(2017, 11, 6)
        # 2018
        self.mark_holiday(2018, 1, 1)
        self.mark_holiday(2018, 1, 2)
        self.mark_holiday(2018, 1, 8)
        self.mark_workday(2018, 4, 28)
        self.mark_workday(2018, 6, 9)
        self.mark_holiday(2018, 11, 5)
        self.mark_workday(2018, 12, 29)
        self.mark_holiday(2018, 12, 31)
        # 2019
        self.mark_holiday(2019, 1, 1)
        self.mark_holiday(2019, 1, 2)
        self.mark_holiday(2019, 1, 7)

