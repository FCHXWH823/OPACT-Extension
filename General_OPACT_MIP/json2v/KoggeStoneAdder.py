from myhdl import *
from BasicModule import *
import sys
import math
@block
def KoggeStone(OUTS,INPUTS,columns):
    width = len(columns);
    inputs_list = [];
    instances_list = [];
    startIndex = 0; endIndex = 0;
    # initialize all inputs
    for i in range(len(columns)):
        col_len = columns[i];
        endIndex = startIndex+col_len;
        tmp_list = [];
        for j in range(startIndex,endIndex):
            tmp_list.append(INPUTS(j));
        inputs_list.append(tmp_list);
        startIndex = endIndex;
    # initial stage: geneate and propagate signals
    dict_init = {};
    for i in range(width):
        GP_Pair = [Signal(intbv(0)[1:]),Signal(intbv(0)[1:])];
        if len(inputs_list[i]) == 1:
            PG_Generator0_ins = PG_Generator0(GP_Pair[0],GP_Pair[1],inputs_list[i][0]);
            instances_list.append(PG_Generator0_ins);
        else:
            PG_Generator1_ins = PG_Generator1(GP_Pair[0],GP_Pair[1],inputs_list[i][0],inputs_list[i][1]);
            instances_list.append(PG_Generator1_ins);
        key = (i,i);
        dict_init[key] = GP_Pair;
    # prefex tree
    dicts = [];
    dicts.append(dict_init);
    nStages = int(math.ceil(math.log2(width)));
    for i in range(nStages):
        dict = {};
        maxNumBits_current = int(pow(2,i+1));
        maxNumBits_last = int(pow(2,i));
        # less than maxNumBits_current
        
        for j in range(maxNumBits_last+1,min(maxNumBits_current,width+1)):
            startIndex = 0;
            endIndex = j - 1;
            startIndex_high = j-maxNumBits_last; endIndex_high = j - 1;
            startIndex_low = 0; endIndex_low = j - 1 - maxNumBits_last;
            GP_Pair_high = dicts[i][(startIndex_high,endIndex_high)];
            iStage = int(math.ceil(math.log2(j-maxNumBits_last)));
            GP_Pair_low = dicts[iStage][(startIndex_low,endIndex_low)];
            if startIndex != 0:
                GP_Pair = [Signal(intbv(0)[1:]),Signal(intbv(0)[1:])];
                BlackCircle_ins = BlackCircle(GP_Pair[0],GP_Pair[1],GP_Pair_high[0],GP_Pair_high[1],GP_Pair_low[0],GP_Pair_low[1]);
                dict[(startIndex,endIndex)] = GP_Pair;
                instances_list.append(BlackCircle_ins);
            else:
                GP_Pair = [Signal(intbv(0)[1:])];
                #GrayCircle_ins = GrayCircle(GP_Pair[0],GP_Pair_high[0],GP_Pair_high[1],GP_Pair_low[0],GP_Pair_low[1]);
                GrayCircle_ins = GrayCircle(GP_Pair[0],GP_Pair_high[0],GP_Pair_high[1],GP_Pair_low[0]);
                dict[(startIndex,endIndex)] = GP_Pair;
                instances_list.append(GrayCircle_ins);
        # equal to maxNumBits_current
        for j in range(0,width-maxNumBits_current+1):
            startIndex = j;
            endIndex = j+maxNumBits_current-1;
            startIndex_high = endIndex - maxNumBits_last + 1; endIndex_high = endIndex;
            startIndex_low = startIndex; endIndex_low = startIndex_high - 1;
            GP_Pair_high = dicts[i][(startIndex_high,endIndex_high)];
            GP_Pair_low = dicts[i][(startIndex_low,endIndex_low)];
            if startIndex != 0:
                GP_Pair = [Signal(intbv(0)[1:]),Signal(intbv(0)[1:])];
                BlackCircle_ins = BlackCircle(GP_Pair[0],GP_Pair[1],GP_Pair_high[0],GP_Pair_high[1],GP_Pair_low[0],GP_Pair_low[1]);
                dict[(startIndex,endIndex)] = GP_Pair;
                instances_list.append(BlackCircle_ins);
            else:
                GP_Pair = [Signal(intbv(0)[1:])];
                #GrayCircle_ins = GrayCircle(GP_Pair[0],GP_Pair_high[0],GP_Pair_high[1],GP_Pair_low[0],GP_Pair_low[1]);
                GrayCircle_ins = GrayCircle(GP_Pair[0],GP_Pair_high[0],GP_Pair_high[1],GP_Pair_low[0]);
                dict[(startIndex,endIndex)] = GP_Pair;
                instances_list.append(GrayCircle_ins);
        dicts.append(dict);
    #final sum
    outs_list = [];
    first_sum = dicts[0][(0,0)][1];
    outs_list.append(first_sum);
    for i in range(1,width):
        sum = Signal(intbv(0)[1:]);
        iStage = int(math.ceil(math.log2(i)));
        FinalSum_ins = FinalSum(sum,dicts[0][(i,i)][1],dicts[iStage][(0,i-1)][0]);
        outs_list.append(sum);
        instances_list.append(FinalSum_ins);
    outs_list.append(dicts[nStages][(0,width-1)][0]);
    outs_vec = ConcatSignal(*reversed(outs_list));
    @always_comb
    def comb():
        OUTS.next = outs_vec;
    return instances_list,comb;

def convert_KoggeStone(hdl,columns):
    width = len(columns);
    Inputs_len = 0;
    for i in range(len(columns)):
        Inputs_len += columns[i];
    INPUTS = Signal(intbv(0)[Inputs_len:]);
    OUTS = Signal(intbv(0)[width+1:]);
    KoggeStone_ins = KoggeStone(OUTS,INPUTS,columns);
    KoggeStone_ins.convert(hdl);

@block
def test_KoggeStone(columns):
    width = len(columns);
    Inputs_len = 0;
    for i in range(len(columns)):
        Inputs_len += columns[i];
    INPUTS = Signal(intbv(0)[Inputs_len:]);
    OUTS = Signal(intbv(0)[width+1:]);
    KoggeStone_ins = KoggeStone(OUTS,INPUTS,columns);
    @instance
    def simu():
        times = 0;
        for i in range(pow(2,Inputs_len)):
            INPUTS.next = intbv(i)[Inputs_len:];
            yield delay(10);
            add_real = 0; add_compute = 0; wrongtime = 0;
            startIndex = 0; endIndex = 0;
            times += 1;
            for j in range(len(columns)):
                endIndex = startIndex + columns[j];
                if columns[j] == 1:
                    add_real += INPUTS[startIndex]*int(pow(2,j));
                else:
                    add_real += (INPUTS[startIndex]+INPUTS[startIndex+1])*int(pow(2,j));
                add_compute += int(pow(2,j))*OUTS[j];
                startIndex = endIndex;
            add_compute += int(pow(2,width))*OUTS[width];
            print("Real sum , Computed sum : %s , %s" % (add_real,add_compute));
            if add_real == add_compute:
                print("Correct!!!");
            else:
                print("Wrong!!!");
                wrongtime += 1;
        print("Total wrong times is %s" % (wrongtime));
        print("Total test times is %s" % (times));
    return instances();


