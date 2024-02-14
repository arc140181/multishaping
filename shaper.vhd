
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;


entity shaper is
	generic ( W : integer := 14;
			   Nx : integer := 4);

    port ( x : in  STD_LOGIC_VECTOR(W-1 downto 0);
	       xen : in STD_LOGIC;
           clk : in  STD_LOGIC;
		   rst : in STD_LOGIC;
           y : out  STD_LOGIC_VECTOR(W-1 downto 0);
		   yen : out STD_LOGIC);
end shaper;

architecture Behavioral of shaper is
	type x_signed is array (0 to Nx-1) of integer range (-(2**(W - 1))) to (2**(W - 1) - 1);

	-- Coefficients
	signal h : x_signed := (others => 0);
	
	-- Block delays
	signal x_z : x_signed := (others => 0);
	
	-- Output signal
	signal y_i : integer := 0;
	
begin
	yen <= (not rst) and xen;
		
	x_p: process(clk)
	begin
		if rising_edge(clk) then
			if rst = '1' then
				for i in 0 to Nx-1 loop
					x_z(i) <= 0;
				end loop;
			elsif xen = '1' then
				x_z(0) <= conv_integer(x);
				for i in 1 to Nx-1 loop
					x_z(i) <= x_z(i-1);
				end loop;
			end if;
		end if;
	end process;
	
	y_p: process(clk)
		variable acc : integer range 0 to 2**(W-1) := 0;
	begin
		if rising_edge(clk) then
			if rst = '1' then
				y_i <= 0;
				for i in 0 to Nx-1 loop   -- Triangular shaping
				    if i < (Nx/2) then
                        h(i) <= i + 1;
                    else
                        h(i) <= Nx - i;
                    end if; 
                end loop;
			else
			    acc := 0;
				for i in 0 to Nx-1 loop
					acc := acc + (h(i) * x_z(i)) / (Nx/2);
				end loop;
				y_i <= acc;
			end if;
		end if;	
	end process;

    process(clk)
    begin
        if falling_edge(clk) then
            if rst = '1' then
                y <= (others => '0');
            else
	            y <= conv_std_logic_vector(y_i, W);
	        end if;
	    end if;
	end process;

end Behavioral;
