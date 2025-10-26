module tb_myhdl_EC32;

wire C;
wire S;
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
        C,
        S
    );
end

myhdl_EC32 dut(
    C,
    S,
    x1,
    x2,
    x3
);

endmodule
