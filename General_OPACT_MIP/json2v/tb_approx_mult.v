module tb_approx_mult;

reg [15:0] A;
reg [15:0] B;
wire [31:0] OUTS;

initial begin
    $from_myhdl(
        A,
        B
    );
    $to_myhdl(
        OUTS
    );
end

approx_mult dut(
    A,
    B,
    OUTS
);

endmodule
