import logging
import math

import re
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def getindex(y, dates):
    """Calculate the index of year in the date list.
    """
    starti = 0
    endi = 0
    for i in range(len(dates)):
        year = dates[i] // 10000
        if year < y:
            starti = i + 1
        elif year == y:
            endi = i
        elif year > y:
            break
    return [starti, endi]

def getmonthindex(m, dates):
    """Calculate the index of month in the date list.
    """
    starti = 0
    endi = 0
    for i in range(len(dates)):
        month = dates[i] // 100
        if month < m:
            starti = i + 1
        elif month == m:
            endi = i
        elif month > m:
            break
    return [starti, endi]

def Dodrawdown(pnlset):
    """Calculate the max drawdown of the pnlset.
    
    Parameters
    ----------
    pnlset : np.array
        The pnlset to be calculated.
    
    Returns
    ----------
    list
        The max drawdown and the start/end index of the drawdown.
    """
    sum = 0
    setst = 0
    start = 0
    dd = 0
    ddstart, ddend = 0, 0

    for i in range(len(pnlset)):
        sum += pnlset[i]
        if setst != 0:
            start = i
            setst = 0 
        if sum >= 0:
            sum = 0
            setst = 1
            start = i
        if sum < dd:
            dd = sum
            ddend = i
            ddstart = start
    return [dd, ddstart, ddend]

def getstats(
        pnl, long, short, ret, sh_hld, sh_trd, poscov, negcov, pospnl, negpnl, IC, inxret
        ):
    """Calculate the statistics of the portfolio.

    Parameters
    ----------
    pnl : np.array
        The pnl of the portfolio.
    long : np.array
        The long size of the portfolio.
    short : np.array
        The short size of the portfolio.
    ret : np.array
        The return of the portfolio.
    sh_hld : np.array
        The holding stocks number of the portfolio.
    sh_trd : np.array
        The trading stocks number of the portfolio.
    poscov : np.array
        The positive coverage of the portfolio.
    negcov : np.array
        The negative coverage of the portfolio.
    pospnl : np.array
        The positive pnl of the portfolio.
    negpnl : np.array
        The negative pnl of the portfolio.
    IC : np.array
        The IC of the single factor.
    inxret : np.array
        The benchmark index return.
    """
    longshort_scale = 1000000
    pnl_scale = 1000000

    ir_all = np.nanmean(ret) / np.nanstd(ret)
    sharpe_all = ir_all * math.sqrt(244) # Average 244 trade days one year
    ret_all = np.nanmean(ret) * 244
    long_all = np.nanmean(long)
    short_all = np.nanmean(short)
    perwin_all = np.nanmean(pnl > 0) # Percentage of winning days
    turnover_all = np.nanmean(np.abs(sh_trd) /2000/10000) #TBD Need to be modified
    poscov_all = np.nanmean(poscov)
    negcov_all = np.nanmean(negcov)
    pospnl_all = np.nanmean(pospnl) / long_all * 244
    negpnl_all = np.nanmean(negpnl) / short_all * -244
    pnl_all = np.nansum(pnl)
    fitness = ir_all * math.sqrt(244) * math.sqrt(abs(ret_all / turnover_all))
    bp_all = pnl_all / np.nansum(sh_trd) * 10000
    drawdown = Dodrawdown(pnl)[0]
    dd_all = drawdown / long_all * (-100)   # Calculate the ratio of maxdown and long position
    ic_all = np.nanmean(IC)
    icir_all = ic_all / np.nanstd(IC)
    ########
    inxret_all = np.nanmean(inxret) * 244
    ########
    ir_all = np.around(ir_all, decimals=3)
    sharpe_all = np.around(sharpe_all, decimals=2)
    ret_all = np.around(ret_all * 100, decimals=2)
    long_all = np.around(long_all / longshort_scale, decimals=2)
    short_all = np.around(short_all / longshort_scale, decimals=2)
    perwin_all = np.around(perwin_all, decimals=2)
    turnover_all = np.around(turnover_all * 100, decimals=2)
    try:
        poscov_all = int(poscov_all)
    except:
        poscov_all = 0
    try:
        negcov_all = int(negcov_all)
    except:
        negcov_all = 0
    pospnl_all = np.around(pospnl_all * 100, decimals=2)
    negpnl_all = np.around(negpnl_all * 100, decimals=2)
    pnl_all = np.around(pnl_all / pnl_scale, decimals=2)
    fitness = np.around(fitness, decimals=2)
    bp_all = np.around(bp_all, decimals=2)
    dd_all = np.around(dd_all, decimals=2)
    ic_all = np.around(ic_all, decimals=2)
    icir_all = np.around(icir_all, decimals=2)
    ########
    inxret_all = np.around(inxret_all * 100, decimals=2)
    ########

    return [
        ir_all, sharpe_all, ret_all, long_all, short_all, perwin_all, turnover_all,
        poscov_all, negcov_all, pospnl_all, negpnl_all, pnl_all, fitness, bp_all,
        dd_all, ic_all, icir_all, inxret_all
        ]

def getrankstats(
            rank1pnl,
            rank2pnl,
            rank3pnl,
            rank4pnl,
            rank5pnl,
            rank6pnl,
            rank7pnl,
            rank8pnl,
            rank9pnl,
            rank10pnl,
            ):
        r1_all = np.nanmean(rank1pnl)
        r2_all = np.nanmean(rank2pnl)
        r3_all = np.nanmean(rank3pnl)
        r4_all = np.nanmean(rank4pnl)
        r5_all = np.nanmean(rank5pnl)
        r6_all = np.nanmean(rank6pnl)
        r7_all = np.nanmean(rank7pnl)
        r8_all = np.nanmean(rank8pnl)
        r9_all = np.nanmean(rank9pnl)
        r10_all = np.nanmean(rank10pnl)

        ir1_all = np.nanmean(rank1pnl) / np.nanstd(rank1pnl)
        ir2_all = np.nanmean(rank2pnl) / np.nanstd(rank2pnl)
        ir3_all = np.nanmean(rank3pnl) / np.nanstd(rank3pnl)
        ir4_all = np.nanmean(rank4pnl) / np.nanstd(rank4pnl)
        ir5_all = np.nanmean(rank5pnl) / np.nanstd(rank5pnl)
        ir6_all = np.nanmean(rank6pnl) / np.nanstd(rank6pnl)
        ir7_all = np.nanmean(rank7pnl) / np.nanstd(rank7pnl)
        ir8_all = np.nanmean(rank8pnl) / np.nanstd(rank8pnl)
        ir9_all = np.nanmean(rank9pnl) / np.nanstd(rank9pnl)
        ir10_all = np.nanmean(rank10pnl) / np.nanstd(rank10pnl)

        ir1_all = np.around(ir1_all, decimals=3)
        ir2_all = np.around(ir2_all, decimals=3)
        ir3_all = np.around(ir3_all, decimals=3)
        ir4_all = np.around(ir4_all, decimals=3)
        ir5_all = np.around(ir5_all, decimals=3)
        ir6_all = np.around(ir6_all, decimals=3)
        ir7_all = np.around(ir7_all, decimals=3)
        ir8_all = np.around(ir8_all, decimals=3)
        ir9_all = np.around(ir9_all, decimals=3)
        ir10_all = np.around(ir10_all, decimals=3)
        r1_all = np.around(r1_all, decimals=2)
        r2_all = np.around(r2_all, decimals=2)
        r3_all = np.around(r3_all, decimals=2)
        r4_all = np.around(r4_all, decimals=2)
        r5_all = np.around(r5_all, decimals=2)
        r6_all = np.around(r6_all, decimals=2)
        r7_all = np.around(r7_all, decimals=2)
        r8_all = np.around(r8_all, decimals=2)
        r9_all = np.around(r9_all, decimals=2)
        r10_all = np.around(r10_all, decimals=2)

        return [
            ir1_all, ir2_all, ir3_all, ir4_all, ir5_all, ir6_all, ir7_all, ir8_all, ir9_all, ir10_all,
            r1_all, r2_all, r3_all, r4_all, r5_all, r6_all, r7_all, r8_all, r9_all, r10_all
        ]

def simsummary(filepath,
               summaryflag= "on",
               group_detail = "on",
               monthly = "off"):
    pnl = []
    long = []
    short = []
    ret = []
    sh_hld = []
    sh_trd = []
    poscov = []
    negcov = []
    pospnl = []
    negpnl = []
    dates = []

    IC = []
    inxret = []

    """Use regular expression to filter factor name"""
    match_result = re.search(r"PNL_(.*?)\.csv", filepath)
    if match_result:
        factor_str = match_result.group(1)
    logger.info(f"Simsummary on  factor: {factor_str}")

    if group_detail == "on":
        rank1pnl = []
        rank2pnl = []
        rank3pnl = []
        rank4pnl = []
        rank5pnl = []
        rank6pnl = []
        rank7pnl = []
        rank8pnl = []
        rank9pnl = []
        rank10pnl = []

    filelines = open(filepath).readlines()
    filelines = filelines[1:]
    for line in filelines:
        line = line.rstrip().split(",")
        if (
            float(line[1]) == 0
            and float(line[2]) == 0
            and float(line[3]) == 0
            and float(line[4]) == 0
            and float(line[5]) == 0
        ):
            continue

        dates.append(int(line[0]))
        pnl.append(float(line[1]))
        long.append(float(line[2]))
        short.append(float(line[3]))
        ret.append(float(line[4]))
        sh_hld.append(float(line[5]))
        sh_trd.append(float(line[6]))
        poscov.append(float(line[7]))
        negcov.append(float(line[8]))
        pospnl.append(float(line[9]))
        negpnl.append(float(line[10]))
        IC.append(float(line[11]))
        ############
        inxret.append(float(line[12]))
        if group_detail == "on":
            rank1pnl.append(float(line[13]))
            rank2pnl.append(float(line[14]))
            rank3pnl.append(float(line[15]))
            rank4pnl.append(float(line[16]))
            rank5pnl.append(float(line[17]))
            rank6pnl.append(float(line[18]))
            rank7pnl.append(float(line[19]))
            rank8pnl.append(float(line[20]))
            rank9pnl.append(float(line[21]))
            rank10pnl.append(float(line[22]))
    # print(len(pnl), len(filelines))
    pnl = np.array(pnl)
    long = np.array(long)
    short = np.array(short)
    ret = np.array(ret)
    sh_hld = np.array(sh_hld)
    sh_trd = np.array(sh_trd)
    poscov = np.array(poscov)
    negcov = np.array(negcov)
    pospnl = np.array(pospnl)
    negpnl = np.array(negpnl)
    IC = np.array(IC)

    inxret = np.array(inxret)

    if group_detail == "on":
        rank1pnl = np.array(rank1pnl)
        rank2pnl = np.array(rank2pnl)
        rank3pnl = np.array(rank3pnl)
        rank4pnl = np.array(rank4pnl)
        rank5pnl = np.array(rank5pnl)
        rank6pnl = np.array(rank6pnl)
        rank7pnl = np.array(rank7pnl)
        rank8pnl = np.array(rank8pnl)
        rank9pnl = np.array(rank9pnl)
        rank10pnl = np.array(rank10pnl)
    
    dates = np.array(dates)
    if len(dates) != 0:
        startdate = dates[0]
        enddate = dates[len(dates) - 1]
        year_s = startdate // 10000
        year_e = enddate // 10000
        yearset = list(range(year_s, year_e + 1, 1))
        startmon = (startdate // 100) % 100
        endmon = (enddate // 100) % 100
        monthset = []
        for y in range(year_s, year_e + 1):
            if y == year_s:
                for m in range(startmon, 13):
                    monthset.append(y * 100 + m)
            elif y == year_e:
                for m in range(1, endmon + 1):
                    monthset.append(y * 100 + m)
            else:
                for m in range(1, 13):
                    monthset.append(y * 100 + m)
    else:
        yearset = []

    headerstr = (
        "dates".ljust(17)
        + " "
        + "long(M)".ljust(7)
        + " "
        + "short(M)".ljust(8)
        + " "
        + "pnl(M)".ljust(7)
        + " "
        + "%ret".ljust(7)
        + " "
        + "tvr".ljust(7)
        + " "
        + "shrp (IR)".ljust(13)
        + " "
        + "%dd".ljust(6)
        + " "
        + "%win".ljust(5)
        + " "
        + "bpmrgn".ljust(13)
        + " "
        + "fitness".ljust(7)
        + " "
        + "Coverage".ljust(13)
        + " "
        + "%posret".ljust(7)
        + " "
        + "%inxret".ljust(7)
        + " "
        + "%negret".ljust(7)
        + " "
        + "IC".ljust(7)
        + " "
        + "ICIR".ljust(7)
    )

    headerstr2 = (
        "\n"
        + "dates".ljust(17)
        + " "
        + "%R1 (IR)".ljust(15)
        + " "
        + "%R2 (IR)".ljust(15)
        + " "
        + "%R3 (IR)".ljust(15)
        + " "
        + "%R4 (IR)".ljust(15)
        + " "
        + "%R5 (IR)".ljust(15)
        + " "
        + "%R6 (IR)".ljust(15)
        + " "
        + "%R7 (IR)".ljust(15)
        + " "
        + "%R8 (IR)".ljust(15)
        + " "
        + "%R9 (IR)".ljust(15)
        + " "
        + "%R10 (IR)".ljust(15)
    )

    resultstr = ""
    resultstr2 = ""

    if monthly == "on":
        yearset = monthset
        year_s = year_s * 100 + startmon
        year_e = year_e * 100 + endmon
    for year in yearset:
        if monthly != "on":
            inx = getindex(year, dates)
        else:
            inx = getmonthindex(year, dates)
        si = inx[0]
        ei = inx[1] + 1
        if year == year_s:
            date_str = str(startdate)
        else:
            if monthly == "on":
                date_str = str(year) + "0101"
            else:
                date_str = str(year) + "0101"
        if year == year_e:
            date_str = date_str + "-" + str(enddate)
        else:
            if monthly == "on":
                date_str = date_str + "-" + str(year) + "1231"
            else:
                date_str = date_str + "-" + str(year) + "1231"

        if summaryflag == "on":
            yearstats = getstats(
                pnl[si:ei],
                long[si:ei],
                short[si:ei],
                ret[si:ei],
                sh_hld[si:ei],
                sh_trd[si:ei],
                poscov[si:ei],
                negcov[si:ei],
                pospnl[si:ei],
                negpnl[si:ei],
                IC[si:ei],
                inxret[si:ei],
            )
            date_str1 = (
                date_str
                + " "
                + str(yearstats[3]).ljust(7)
                + " "
                + str(yearstats[4]).ljust(8)
                + " "
                + str(yearstats[11]).ljust(7)
                + " "
                + str(yearstats[2]).ljust(7)
                + " "
                + str(yearstats[6]).ljust(7)
                + " "
                + str(str(yearstats[1]) + "(" + str(yearstats[0]) + ")").ljust(13)
                + " "
                + str(yearstats[14]).ljust(6)
                + " "
                + str(yearstats[5]).ljust(5)
                + " "
                + str(yearstats[13]).ljust(13)
                + " "
                + str(yearstats[12]).ljust(7)
                + " "
                + str(str(yearstats[7]) + " X " + str(yearstats[8])).ljust(13)
                + " "
                + str(yearstats[9]).ljust(7)
                + " "
                + str(yearstats[17]).ljust(7)
                + " "
                + str(yearstats[10]).ljust(7)
                + " "
                + str(yearstats[15]).ljust(7)
                + " "
                + str(yearstats[16]).ljust(7)
            )
            resultstr = resultstr + date_str1 + "\n"

        if group_detail == "on":
            yearstats2 = getrankstats(
                rank1pnl[si:ei],
                rank2pnl[si:ei],
                rank3pnl[si:ei],
                rank4pnl[si:ei],
                rank5pnl[si:ei],
                rank6pnl[si:ei],
                rank7pnl[si:ei],
                rank8pnl[si:ei],
                rank9pnl[si:ei],
                rank10pnl[si:ei],
            )
            date_str2 = (
                date_str
                + " "
                + str(str(yearstats2[0]) + "(" + str(yearstats2[10]) + ")").ljust(15)
                + str(str(yearstats2[1]) + "(" + str(yearstats2[11]) + ")").ljust(15)
                + str(str(yearstats2[2]) + "(" + str(yearstats2[12]) + ")").ljust(15)
                + str(str(yearstats2[3]) + "(" + str(yearstats2[13]) + ")").ljust(15)
                + str(str(yearstats2[4]) + "(" + str(yearstats2[14]) + ")").ljust(15)
                + str(str(yearstats2[5]) + "(" + str(yearstats2[15]) + ")").ljust(15)
                + str(str(yearstats2[6]) + "(" + str(yearstats2[16]) + ")").ljust(15)
                + str(str(yearstats2[7]) + "(" + str(yearstats2[17]) + ")").ljust(15)
                + str(str(yearstats2[8]) + "(" + str(yearstats2[18]) + ")").ljust(15)
                + str(str(yearstats2[9]) + "(" + str(yearstats2[19]) + ")").ljust(15)
            )
            resultstr2 = resultstr2 + date_str2 + "\n"

    if summaryflag == "on":
        if len(dates) > 0:
            yearstats = getstats(
                pnl,
                long,
                short,
                ret,
                sh_hld,
                sh_trd,
                poscov,
                negcov,
                pospnl,
                negpnl,
                IC,
                inxret,
            )
            date_str = str(startdate)
            date_str = date_str + "-" + str(enddate)
            date_str = (
                date_str
                + " "
                + str(yearstats[3]).ljust(7)
                + " "
                + str(yearstats[4]).ljust(8)
                + " "
                + str(yearstats[11]).ljust(7)
                + " "
                + str(yearstats[2]).ljust(7)
                + " "
                + str(yearstats[6]).ljust(7)
                + " "
                + str(str(yearstats[1]) + "(" + str(yearstats[0]) + ")").ljust(13)
                + " "
                + str(yearstats[14]).ljust(6)
                + " "
                + str(yearstats[5]).ljust(5)
                + " "
                + str(yearstats[13]).ljust(6)
                + " "
                + str(yearstats[12]).ljust(7)
                + " "
                + str(str(yearstats[7]) + " X " + str(yearstats[8])).ljust(13)
                + " "
                + str(yearstats[9]).ljust(7)
                + " "
                + str(yearstats[17]).ljust(7)
                + " "
                + str(yearstats[10]).ljust(7)
                + " "
                + str(yearstats[15]).ljust(7)
                + " "
                + str(yearstats[16]).ljust(7)
            )
            resultstr = resultstr +"\n" + date_str + "\n"
        logger.info(f"\n{headerstr}\n{resultstr}")

    if group_detail == "on":
        if len(dates) > 0:
            yearstats2 = getrankstats(
                rank1pnl,
                rank2pnl,
                rank3pnl,
                rank4pnl,
                rank5pnl,
                rank6pnl,
                rank7pnl,
                rank8pnl,
                rank9pnl,
                rank10pnl,
            )
            date_str2 = str(startdate)
            date_str2 = date_str + "-" + str(enddate)
            date_str2 = (
                date_str2
                + " "
                + str(str(yearstats2[0]) + "(" + str(yearstats2[10]) + ")").ljust(15)
                + str(str(yearstats2[1]) + "(" + str(yearstats2[11]) + ")").ljust(15)
                + str(str(yearstats2[2]) + "(" + str(yearstats2[12]) + ")").ljust(15)
                + str(str(yearstats2[3]) + "(" + str(yearstats2[13]) + ")").ljust(15)
                + str(str(yearstats2[4]) + "(" + str(yearstats2[14]) + ")").ljust(15)
                + str(str(yearstats2[5]) + "(" + str(yearstats2[15]) + ")").ljust(15)
                + str(str(yearstats2[6]) + "(" + str(yearstats2[16]) + ")").ljust(15)
                + str(str(yearstats2[7]) + "(" + str(yearstats2[17]) + ")").ljust(15)
                + str(str(yearstats2[8]) + "(" + str(yearstats2[18]) + ")").ljust(15)
                + str(str(yearstats2[9]) + "(" + str(yearstats2[19]) + ")").ljust(15)
            )
            resultstr2 = resultstr2 + date_str2 + "\n"
        logger.info(f"\n{headerstr2}\n{resultstr2}")
    
    return f"\n{headerstr}\n{resultstr}"




    
        