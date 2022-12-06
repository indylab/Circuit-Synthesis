 circuit: LNA
.include {model_path}
.option TEMP=27C


mn2 net1 vdd net3 0 NMOS w=75.0u l=45.0n 
mn1 net3 net5 net9 0 NMOS w=75.0u l=45.0n 
mn3 net4 net4 0 0 NMOS w=6.5u l=45.0n 
R3 vdd net1 400 
RBIAS net4 net6 2.5K 
RREF vdd net4 1.5K 
Lg net6 net5 11n 
Ls 0 net9 750p 
Ld vdd net1 4n 
C7 net1 net8 10n 
C5 net8 0 400f 
CB net7 net6 10n 
C3 net5 net9 300f 
VDD vdd 0 dc 1.2 
V2 net8 0 dc 0 ac 1 portnum 2 z0 50
V1 net7 0 dc 0 ac 1 portnum 1 z0 50


.control

set ls_array = ( {LNA-ls_array} )
set ld_array = ( {LNA-ld_array} )
set lg_array = ( {LNA-lg_array} )
set w_array = ( {LNA-w_array} )

set i = {num_samples}
let index = 1
repeat $i
    alter Ls = $ls_array[$&index]
    alter Ld = $ld_array[$&index]
    alter Lg = $lg_array[$&index]
    alter @mn2[w] = $w_array[$&index]
    alter @mn1[w] = $w_array[$&index]
    sp lin 100 10M 10G 1
    let s21 = abs(S_2_1)
    let s12 = abs(S_1_2)
    let s11 = abs(S_1_1)
    let s11db = 10*log(s11)
    let reals11 = minimum((S11db))

    let G1 = (real(s21))^2
    let G2 = vecmax(G1)
    print G2 >> {out}/LNA-Gt.csv


    print reals11 >> {out}/LNA-s11.csv
    let nf = real(minimum(NF))
    print nf >> {out}/LNA-nf.csv

    print $ls_array[$&index] >> {out}/LNA-ls.csv
    print $ld_array[$&index] >> {out}/LNA-ld.csv
    print $lg_array[$&index] >> {out}/LNA-lg.csv
    print $w_array[$&index] >> {out}/LNA-w.csv

    let index = index + 1
end
.endc
.end