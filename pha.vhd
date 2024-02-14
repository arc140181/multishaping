library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;


entity pha is
    generic ( W : integer := 14);
    port ( clk : in STD_LOGIC;
           rst : in STD_LOGIC;
           y : out STD_LOGIC_VECTOR (W-1 downto 0);
           yen : out STD_LOGIC;
           last : out STD_LOGIC_VECTOR (W-1 downto 0);
           xen : in STD_LOGIC;
           x : in STD_LOGIC_VECTOR (W-1 downto 0));
end pha;

architecture Behavioral of pha is
    signal x_z : integer := 0;
    signal flag0, flag1 : std_logic;
    signal ph : integer := 0;
    constant THR : integer := 100;
begin

	x_p: process(clk)
	begin
		if rising_edge(clk) then
			if rst = '1' then
				x_z <= 0;
		    elsif xen = '1' then
			    x_z <= conv_integer(x);
			end if;
		end if;
	end process;

    flags_p: process(clk)
    begin
        if falling_edge(clk) then
            y <= (others => '0');
            yen <= '0';
            if rst = '1' then
                flag0 <= '0';
                flag1 <= '0';
                last <= (others => '0');
                ph <= 0;
            elsif flag0 = '0' and flag1 = '1' then
                last <= conv_std_logic_vector(ph, W);
                if x_z < THR then
                    flag0 <= '0';
                    flag1 <= '0';
                else
                    flag0 <= '0';
                    flag1 <= '1';
                end if;
            elsif flag0 = '0' and flag1 = '0' then
                ph <= 0;
                if x_z > THR then
                    flag0 <= '1';
                    flag1 <= '0';
                end if;
            else --if flag0 = '1' and flag1 = '0' then
                if x_z > ph then
                    ph <= x_z;
                    flag0 <= '1';
                    flag1 <= '0';
                elsif x_z < (ph / 2) then
                    flag0 <= '0';
                    flag1 <= '1';
                    y <= conv_std_logic_vector(ph, W);
                    yen <= '1';
                end if;                    
            end if;
        end if;
    end process;
    

end Behavioral;
