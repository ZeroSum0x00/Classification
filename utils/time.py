from datetime import datetime, timedelta



def datetime2string(object):
    if not isinstance(object, timedelta):
        raise TypeError("input must be datetime.timedelta instance")
        
    seconds = int(object.total_seconds())
    periods = [
        ("year",   60*60*24*365),
        ("month",  60*60*24*30),
        ("day",    60*60*24),
        ("hour",   60*60),
        ("minute", 60),
        ("second", 1)
    ]

    strings=[]
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value , seconds = divmod(seconds, period_seconds)
            has_s = "s" if period_value > 1 else ""
            strings.append("%s %s%s" % (period_value, period_name, has_s))

    return ", ".join(strings)
