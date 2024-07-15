# Multishaping
VHDL code for Multishaping algorithm. Created with Xilinx Vivado and exportable to any architecture with enough resources to host this project.

- *shaper.vhd*. Finite Impulse Respose (FIR) shaper.
- *pha.vhd*. Pulse height analiser.
- *fsm.vhd*. Finite state machine.
- *mshape.vhd*. The entire triplicated detection chain.
- *mshape_tb.vhd*. Testbench of the detection chain.
- *signal_delta_100.txt*. Example pulses.

The Python simulation is located in *python* folder. It just need Numpy and Matplotlib libraries.

***

For this project, tools from the Xilinx family were employed to synthesize the design in a Spartan-7 FPGA, specifically a XC7S75FGGA676-1 device. The implementation was executed using generic VHDL without relying on any specific component of the device. As a result, synthesizing this design in any other FPGA from a different manufacturer should not pose any issues.

![SchDig](https://github.com/user-attachments/assets/1105c1e9-defa-4506-b631-6287d05f62c5)

<img src="https://github.com/user-attachments/assets/1105c1e9-defa-4506-b631-6287d05f62c5" alt="HykI4IXXA" width="500"/>

*Figure 1. Proposed scheme of implementation of the presented method in VHDL with its corresponding interfaces. CLK and RST connections and Shaper 1, Shaper 2, PHA 1 and PHA 2 are omitted for clarity.*

All the interfaces have in common the resolution in bytes *W* of the signals. Apart from signals *CLK* (clock signal) and *RST* (reset signal), which are common for all the proposed interfaces, in the case of the shaper, the input signals are: *X*: input signal from the ADC; *XEN*: input signal enable. The output signals are: *Y*: shaped signal; *YEN*: shaped signal valid. The shaping type and duration are configured internally.

The input signals of the PHA are: *X*: input signal from shapers; *XEN*: input signal enable. The output signals are: *Y*: pulse height (zero if not pulse detected); *YEN*: pulse height detected; *LAST*: height of the last pulse detected. The threshold level (see Section \ref{Method}) is a parameter of this component selected during synthesis.

Finally the input signals of the FSM are: *IN0, IN1, IN2*: pulse heights detected; *IN0EN, IN1EN, IN2EN*: pulse heights detected valid; *LAST0, LAST1, LAST2*: height of the last pulses detected by PHAs. The output signals are: *Y*: pulse height (zero if not pulse height detected); *YEN*: pulse height detected.

The following table enumerates the components used for *W*=14 in a Xilinx XC7S75FGGA676-1. This FPGA is one with more limited resources within the Spartan-7 family. However, the proposed design fits perfectly. Note that specific resources from this architecture or from Xilinx, such as Digital Signal Processors (DSPs), were not used. Instead, generic resources have been employed to make the design transferable to other programmable devices, even those different from those provided by Xilinx. If an FPGA with more resources had been used, and specific resources had been employed, it would allow replicating the proposed method for multiple detectors using a single device.

|     | Resource | Number |
|-----|----------|--------|
| Slice LUTs (logic) | 24942 | 51.96% |
| Slice registers | 695 | 0.72% |
| Other resources | 0 | 0.0% |

