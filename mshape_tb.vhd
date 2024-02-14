
LIBRARY ieee;
USE ieee.std_logic_1164.all;
USE ieee.std_logic_unsigned.all;
USE ieee.numeric_std.all;
USE ieee.math_real.all;
USE ieee.std_logic_arith.all;
USE ieee.std_logic_textio.all;
USE std.textio.ALL;
 
entity mshape_tb is
--  Port ( );
end mshape_tb;
 
architecture Behavioral of mshape_tb is
 
    -- Component Declaration for the Unit Under Test (UUT)
     
    component mshape
	generic ( W : integer := 14);
    port ( clk : in STD_LOGIC;
           rst : in STD_LOGIC;
           x : in STD_LOGIC_VECTOR (W-1 downto 0);
           xen : in STD_LOGIC;
           y : out STD_LOGIC_VECTOR (W-1 downto 0);
           yen : out STD_LOGIC);
    end component;


   --Inputs
   signal x : std_logic_vector(13 downto 0) := (others => '0');
   signal xen : std_logic := '0';
   signal clk : std_logic := '0';
   signal rst : std_logic := '0';

 	--Outputs
   signal yen : std_logic;
   signal y : std_logic_vector(13 downto 0);

   -- Clock period definitions
   --constant clk_period : time := 20 ns;
   constant clk_period : time := 10 ns;    -- ML410
	
   signal A : real := 1.0 * (2**13);	-- Signal amplitude
   signal An : real := 500.0;
 
BEGIN
 
	-- Instantiate the Unit Under Test (UUT)
   uut: mshape
--   GENERIC MAP (
--          W <= 14
--   )
   PORT MAP (
          x => x,
          xen => '1',
          clk => clk,
          rst => rst,
          y => y,
          yen => yen
        );

-- Clock process definitions
   clk_process :process
   begin
		clk <= '0';
		wait for clk_period/2;
		clk <= '1';
		wait for clk_period/2;
   end process;
 
	-- Read signals from text file
	read_file: process
		file my_file : text open read_mode is "D:/proj/xilinx/multishape_1/multishape_1.srcs/sim_1/new/signal_delta_100.txt";
--		file my_file : text open read_mode is "signal_rc_100.txt";
		variable my_line : line;
		variable t_tmp : real;
		variable t_tmp_ant : real;
		variable counter : integer;
		variable y_tmp : real;
		variable sig_in_tmp, sig_in_tmp2 : integer;
		variable time_interval_ns : integer;
		variable seed1, seed2 : positive;
		variable rand : real;

	begin
        rst <= '1';
        wait for 50 ns;
        rst <= '0';
        wait for 100 ns;
		
		t_tmp_ant := 0.0;
		
		readline(my_file, my_line);
		while not endfile(my_file) loop
			
			readline(my_file,my_line); 
			read(my_line, t_tmp);
			read(my_line, y_tmp);
			
			-- Noise
			UNIFORM(seed1, seed2, rand);
			
--			sig_in_tmp := integer((y_tmp * A) + ((rand * An)-(An/2.0)));
			sig_in_tmp := integer(y_tmp * A);
			
			x <= conv_std_logic_vector(sig_in_tmp, 14);
			
			wait for clk_period;
			
		end loop;
		file_close(my_file);
		
		wait;
	end process;
	
	process
	begin
	   xen <= '0';
	   wait for 2 us;
	   xen <= '1';
	   wait;
	end process;

end Behavioral;
