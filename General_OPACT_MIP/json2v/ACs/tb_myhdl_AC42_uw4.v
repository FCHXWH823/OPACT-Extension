module tb_myhdl_AC42_uw4;

wire C;
wire S;
reg x1;
reg x2;
reg x3;
reg x4;

initial begin
    $from_myhdl(
        x1,
        x2,
        x3,
        x4
    );
    $to_myhdl(
        C,
        S
    );
end

myhdl_AC42_uw4 dut(
    C,
    S,
    x1,
    x2,
    x3,
    x4
);

endmodule
