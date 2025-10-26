from myhdl import *
import random
import math

@block
def AND(out,a,b):
    @always_comb
    def comb():
        out.next = a & b;
    return comb;


@block
def FA(cout,sum,a,b,c):
    @always_comb
    def comb():
        sum.next = a^b^c;
        cout.next = a&b | a&c | b&c;
    return comb;

@block
def test_FA():
    cout,sum,a,b,c = [Signal(intbv(0)[1:]) for i in range(5)];
    FA_1 = FA(cout,sum,a,b,c);
    @instance
    def simu():
        print("cout sum a b c");
        for i in range(pow(2,3)):
            bv = intbv(i)[3:];
            a.next,b.next,c.next = [bv[j] for j in range(3)];
            yield delay(10);
            print("%s %s %s %s %s" % (cout,sum,a,b,c));
    return instances();

def convert_FA(hdl):
    cout,sum,a,b,c = [Signal(intbv(0)[1:]) for i in range(5)];
    FA_1 = FA(cout,sum,a,b,c);
    FA_1.convert(hdl);

@block
def HA(cout,sum,a,b):
    @always_comb
    def comb():
        sum.next = a^b;
        cout.next = a&b;
    return comb;

@block
def test_HA():
    cout,sum,a,b = [Signal(intbv(0)[1:]) for i in range(4)];
    HA_1 = HA(cout,sum,a,b);
    @instance
    def simu():
        print("cout sum a b");
        for i in range(pow(2,2)):
            bv = intbv(i)[2:];
            a.next,b.next = [bv[1],bv[0]];
            yield delay(10);
            print("%s %s %s %s" % (cout,sum,a,b));

    return instances();

def convert_HA(hdl):
    cout,sum,a,b = [Signal(intbv(0)[1:]) for i in range(4)];
    HA_1 = HA(cout,sum,a,b);
    HA_1.convert(hdl);

@block
def AC42(out1,out2,p0,p1,p2,p3):
    @always_comb
    def comb():
        out2.next = p0&p1 | p2 | p3;
        out1.next = p2&p3 | p0 | p1;
    return comb;

@block
def test_AC42():
    out1,out2,p0,p1,p2,p3 = [Signal(intbv(0)[1:]) for i in range(6)];
    AC42_1 = AC42(out1,out2,p0,p1,p2,p3);
    @instance
    def simu():
        print("p3 p2 p1 p0 out2 out1");
        for i in range(pow(2,4)):
            bv = intbv(i)[4:];
            p0.next,p1.next,p2.next,p3.next = [bv[j] for j in range(4)];
            yield delay(10);
            print("%s %s %s %s %s %s" % (p3,p2,p1,p0,out2,out1));
    return instances();

def convert_AC42(hdl):
    out1,out2,p0,p1,p2,p3 = [Signal(intbv(0)[1:]) for i in range(6)];
    AC42_1 = AC42(out1,out2,p0,p1,p2,p3);
    AC42_1.convert(hdl);

@block
def slice_shadower(OUTS,INPUTS):
    @always_comb
    def comb():
        OUTS.next = INPUTS;
    return comb;

@block
def AC32(out1,out2,p0,p1,p2):
    @always_comb
    def comb():
        out2.next = p0 & p1 | p2;
        out1.next = p0 | p1;
    return comb;

@block
def test_AC32():
    out1,out2,p0,p1,p2 = [Signal(intbv(0)[1:]) for i in range(5)];
    AC32_1 = AC32(out1,out2,p0,p1,p2);
    @instance
    def simu():
        print("p2 p1 p0 out2 out1");
        for i in range(pow(2,3)):
            bv = intbv(i)[3:];
            p0.next,p1.next,p2.next = [bv[j] for j in range(3)];
            yield delay(10);
            print("%s %s %s %s %s" % (p2,p1,p0,out2,out1));
    return instances();

def convert_AC32():
    out1,out2,p0,p1,p2 = [Signal(intbv(0)[1:]) for i in range(5)];
    AC32_1 = AC32(out1,out2,p0,p1,p2);
    AC32_1.convert(hdl);


@block
def AC53(out1,out2,out3,p0,p1,p2,p3,p4):
    @always_comb
    def comb():
        out3.next = (p0 & p1) | p2 | p3;
        out2.next = p0 | p1;
        out1.next = (p2 & p3) | p4;
    return comb;

@block
def AC63(out1,out2,out3,p0,p1,p2,p3,p4,p5):
    @always_comb
    def comb():
        out3.next = (p0 & p1) | p2 | p3;
        out2.next = (p2 & p3) | p4 | p5;
        out1.next = (p4 & p5) | p0 | p1;
    return comb;

@block
def PG_Generator0(g,p,a):
    @always_comb
    def comb():
        p.next = a;
        g.next = intbv(0)[1:];
    return comb;

@block
def PG_Generator1(g,p,a,b):
    @always_comb
    def comb():
        p.next = a^b;
        g.next = a&b;
    return comb;

@block
def BlackCircle(go,po,gcurrent,pcurrent,gnext,pnext):
    @always_comb
    def comb():
        po.next = pcurrent & pnext;
        go.next = gcurrent | (pcurrent & gnext);
    return comb;

@block
def GrayCircle(go,gcurrent,pcurrent,gnext):
    @always_comb
    def comb():
        go.next = gcurrent | (pcurrent & gnext);
    return comb;

@block
def FinalSum(sum,p,c):
    @always_comb
    def comb():
        sum.next = p^c;
    return comb;