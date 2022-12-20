circuit: PA
.include {model_path}
.option TEMP=27C
V9 VinBias2 0  dc 750m
V8 Vbias2 0 dc 1.2
V3 Vdd 0  dc 1.2
V7 VinBias 0 dc 800m
V0 Vbias 0 dc 1.2
R0 net14 net10 2
R1 Vin1 net14  25
R2 Vin2 net15  25
K910  L9 L10 1
K811  L8 L11 1
K07  L0 L7 1
K16  L1 L6 1
K35  L3 L5 1
K24  L2 L4 1
L15 net16 net1 175p
L14 net15 net2 175p
L13 net8 net3 175p
L12 net9 net4 175p
L11 net10 Vout2n 60p 
L10 net14 Vout2p 60p 
L9 Vdd net11 475p  
L8 Vdd net12 475p
L7 VinBias2 Vinp2 310p 
L6 VinBias2 Vinn2 310p 
L0 Vdd net6 460p
L1 Vdd net7 460p
L5 Vinn VinBias 80p 
L4 Vinp VinBias 80p 
L3 net14 net13 350p 
L2 net15 net13 350p 
V11 Vin1 Vin2 sin(0 350m 28G) 
E1 (vout 0 Vout2p Vout2n) 1
R8 vout 0 50
mn8 net12 Vbias2 net9 0 NMOS w=32u l=45.0n
mn7 net11 Vbias2 net8 0 NMOS w=32u l=45.0n 
mn6 net4 Vinn2 0 0 NMOS w=32u l=45.0n 
mn5 net3 Vinp2 0 0 NMOS w=32u l=45.0n 
mn3 net1 Vinn 0 0 NMOS w=22u l=45.0n 
mn2 net7 Vbias net16 0 NMOS w=22u l=45.0n 
mn1 net2 Vinp 0 0 NMOS w=22u l=45.0n 
mn0 net6 Vbias net15 0 NMOS w=22u l=45.0n 
.control
set lint_array = ( {pa-lint_array} )
set ls1_array = ( {pa-ls1_array} )
set ls2_array = ( {pa-ls2_array} )
set vb1_array = ( {pa-vb1_array} )
set vb2_array = ( {pa-vb2_array} )
set w1_array = ( {pa-w1_array} )
set w2_array = ( {pa-w2_array} )
set i = {num_samples}
let index = 1
repeat $i
alter L12 = $lint_array[$&index]
alter L13 = $lint_array[$&index]
alter L14 = $lint_array[$&index]
alter L15 = $lint_array[$&index]
 
alter L0 = $ls1_array[$&index]
alter L1 = $ls1_array[$&index]
alter L8 = $ls2_array[$&index]
alter L9 = $ls2_array[$&index]
 
alter @V7[dc] = $vb1_array[$&index]
      
alter @V9[dc] = $vb2_array[$&index]
       
alter @mn0[w] = $w1_array[$&index]
alter @mn1[w] = $w1_array[$&index]
alter @mn2[w] = $w1_array[$&index]
alter @mn3[w] = $w1_array[$&index]
alter @mn5[w] = $w2_array[$&index]
alter @mn6[w] = $w2_array[$&index]
alter @mn7[w] = $w2_array[$&index]
alter @mn8[w] = $w2_array[$&index]
tran 0.1n 10n
let vin= v(Vin1) - v(Vin2)
*plot v(vout)
*plot vin
*plot i(V11)
*plot i(E1)
*plot i(V3)
let Pout = maximum(v(vout)) * maximum(i(E1))
let Pin = maximum(i(V11)) * maximum(vin)
let G = 10*log (Pout/Pin)
print G >> {out}/pa-gain1.csv
meas tran id3 RMS i(V3) from = 5ns to = 8ns
print id3
let Psupp= id3 * 1.2
let PAE =  100 * (Pout - Pin)/Psupp
print PAE >> {out}/pa-PAE1.csv
let DE =100 * Pout/Psupp
print DE >> {out}/pa-DE1.csv 
let lint = $lint_array[$&index]
let vb1 = $vb1_array[$&index]
let vb2 = $vb2_array[$&index]
let ls2 = $ls2_array[$&index]
let w1 = $w1_array[$&index]
let w2 = $w2_array[$&index]
let ls1 = $ls1_array[$&index]
print  w2 >> {out}/pa-w2.csv 
print w1 >> {out}/pa-w1.csv
print vb2 >> {out}/pa-vb2.csv
print vb1 >> {out}/pa-vb1.csv
print ls2 >> {out}/pa-ls2.csv
print  ls1 >> {out}/pa-ls1.csv
print lint >> {out}/pa-lint.csv
let index = index + 1
end
.endc
.end