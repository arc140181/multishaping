
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;


entity mshape is
	generic ( W : integer := 14);
    port ( clk : in STD_LOGIC;
           rst : in STD_LOGIC;
           x : in STD_LOGIC_VECTOR (W-1 downto 0);
           xen : in STD_LOGIC;
           y : out STD_LOGIC_VECTOR (W-1 downto 0);
           yen : out STD_LOGIC);
end mshape;

architecture Behavioral of mshape is

    component shaper
        generic ( W : integer := 14;
                   Nx : integer := 4);
        port ( x : in  STD_LOGIC_VECTOR(W-1 downto 0);
               xen : in STD_LOGIC;
               clk : in  STD_LOGIC;
               rst : in STD_LOGIC;
               y : out  STD_LOGIC_VECTOR(W-1 downto 0);
               yen : out STD_LOGIC);
    end component;
    
    component pha
        generic ( W : integer := 14);
        port ( clk : in STD_LOGIC;
               rst : in STD_LOGIC;
               y : out STD_LOGIC_VECTOR (W-1 downto 0);
               yen : out STD_LOGIC;
               last : out STD_LOGIC_VECTOR (W-1 downto 0);
               xen : in STD_LOGIC;
               x : in STD_LOGIC_VECTOR (W-1 downto 0));
    end component;
    
    component fsm is
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
    end component;

    type arr_reg is array (0 to 2) of std_logic_vector(W-1 downto 0);
    type arr_reg_val is array (0 to 2) of std_logic;
    type arr_reg_int is array (0 to 2) of integer;
    
    constant PULSELEN : arr_reg_int := (10, 20, 40);
        
    signal shaped : arr_reg;
    signal shaped_val : arr_reg_val;
    signal ph : arr_reg;
    signal ph_val : arr_reg_val;
    signal last : arr_reg;
    
begin

    m: for i in 0 to 2 generate
        shaper_inst : shaper
        generic map (
           W => W,
           Nx => PULSELEN(i)
        )
        port map (
           x => x,
           xen => xen,
           clk => clk,
           rst => rst,
           y => shaped(i),
           yen => shaped_val(i)
        );
        
        pha_inst : pha
        generic map (
           W => W
        )
        port map (
           clk => clk,
           rst => rst,
           y => ph(i),
           yen => ph_val(i),
           last => last(i),
           xen => shaped_val(i),
           x => shaped(i)
        );
    end generate;
    
    fsm_m : fsm
        generic map (
           W => W,
           PERC => 20
        )
        port map (
           in0 => ph(0),
           last0 => last(0),
           in1 => ph(1),
           last1 => last(1),
           in2 => ph(2),
           last2 => last(2),
           clk => clk,
           rst => rst,
           y => y,
           yen => yen
        );

end Behavioral;
