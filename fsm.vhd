
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity fsm is
    generic ( W : integer := 14;
              PERC : integer := 10
    
    );
    port ( in0 : in STD_LOGIC_VECTOR (W-1 downto 0);
           last0 : in STD_LOGIC_VECTOR (W-1 downto 0);
           in1 : in STD_LOGIC_VECTOR (W-1 downto 0);
           last1 : in STD_LOGIC_VECTOR (W-1 downto 0);
           in2 : in STD_LOGIC_VECTOR (W-1 downto 0);
           last2 : in STD_LOGIC_VECTOR (W-1 downto 0);
           
           y : out STD_LOGIC_VECTOR (W-1 downto 0);
           yen : out STD_LOGIC;
    
           clk : in STD_LOGIC;
           rst : in STD_LOGIC);
end fsm;

architecture Behavioral of fsm is
    signal active : integer range 0 to 2;
begin
    process(clk)
        variable last_a : integer range 0 to 2**(W-1);
        variable last_b : integer range 0 to 2**(W-1);
    begin
        if rising_edge(clk) then
            y <= (others => '0');
            yen <= '0';
            if rst = '1' then
                active <= 0;
            elsif active = 0 and conv_integer(in0) > 0 then
                active <= 1;
            elsif active = 1 and conv_integer(in0) > 0 then
                y <= last0;
                yen <= '1';
                active <= 1;
            elsif active = 1 and conv_integer(in1) > 0 then
                active <= 2;
            elsif active = 1 and conv_integer(in2) > 0 then
                y <= last0;
                yen <= '1';
                active <= 0;
            elsif active = 2 and (conv_integer(in0) > 0 or conv_integer(in1) > 0) then
                last_a := conv_integer(last0);
                last_b := conv_integer(last1);
                
                if abs(last_a - last_b) > PERC then
                    y <= last0; -- Possible pile-up in channel 1
                else
                    y <= last1;
                end if;
                yen <= '1';
                active <= 1;
            elsif active = 2 and conv_integer(in2) > 0 then
                last_a := conv_integer(in2);
                last_b := conv_integer(last1);
                
                if abs(last_a - last_b) > PERC then
                    y <= last1; -- Possible pile-up in channel 1
                else
                    y <= in2;
                end if;
                yen <= '1';
                active <= 0;
            end if;
        end if;
    end process;
end Behavioral;
