module tb_myhdl_EC22;

wire C;
wire S;
reg x1;
reg x2;

initial begin
    $from_myhdl(
        x1,
        x2
    );
    $to_myhdl(
        C,
        S
    );
end

myhdl_EC22 dut(
    C,
    S,
    x1,
    x2
);

endmodule
