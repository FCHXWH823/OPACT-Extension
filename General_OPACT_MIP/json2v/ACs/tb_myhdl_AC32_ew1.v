module tb_myhdl_AC32_ew1;

wire w2;
wire w1;
reg x1;
reg x2;
reg x3;

initial begin
    $from_myhdl(
        x1,
        x2,
        x3
    );
    $to_myhdl(
        w2,
        w1
    );
end

myhdl_AC32_ew1 dut(
    w2,
    w1,
    x1,
    x2,
    x3
);

endmodule
