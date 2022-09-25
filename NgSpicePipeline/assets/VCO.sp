nmos:VCO

.include {model_path}

.option TEMP=27C

L2 net16 net18 590.00p
L0 0 ouputn 590.00p
L1 0 ouputp 590.00p
mn3 net1 net1 vdd! 0 PMOS w=2u l=1u
C6 net16 net18 400f ic=200.0m
C0 ouputn 0 80f
C1 ouputp 0 80f ic=200.0m
V2 vctrl 0 dc 0
V0 vdd! 0 dc 1.2
mp2 net18 net1 vdd! vdd! PMOS w=100u l=1u
mp1 ouputp ouputn net16 vdd! PMOS w=4u l=65n
mp0 ouputn ouputp net16 vdd! PMOS w=4u l=65n
R0 0 net1 5.0K
mp4 ouputn vctrl ouputn ouputn PMOS w=15u l=65n
mp5 ouputp vctrl ouputp ouputp PMOS w=15u l=65n

.control
set w_array = ( {w_array} )
set w1_array = ( {w1_array} )
set w2_array = ( {w2_array} )

set i = {num_samples}
let index = 1
repeat $i
    alter @mp1[w] = w_array[$&index]
    alter @mp0[w] = w_array[$&index]
    alter @mp5[w]= w2_array[$&index]
    alter @mp4[w]= w2_array[$&index]
    alter @mn3[w] = w1_array[$&index]

    op
    let gm = @mp1[gm]
    let rs = (590p*20G)/15
    let vn = (1.38e-23)*300*gm*15*rs*rs*10k
    let ib = @mp2[id]
    let Va =  ib*15*15*rs
    let Pnoise = (8*vn)/(Va*Va)
    let dbpn = db(Pnoise)
    let pw = ib*1.2
    print pw >> {out}power-test.csv
    print dbpn >> {out}pnoise-test.csv

    tran 1ns 100ns
    meas tran tdiff1 TRIG v(ouputn) VAL=0 RISE=1 TARG v(ouputn) VAL=0 RISE=2
    let f1 = 1/tdiff1
    alter V2 dc = 1
    tran 1ns 100ns
    meas tran tdiff2 TRIG v(ouputn) VAL=0 RISE=1 TARG v(ouputn) VAL=0 RISE=2
    let f2 = 1/tdiff2
    let ft = abs(tran1.f1-f2)
    print ft >> {out}tuningrange-test.csv

    print $w_array[$&index] >> {out}w-test.csv
    print $w1_array[$&index] >> {out}w1-test.csv
    print $w2_array[$&index] >> {out}w2-test.csv

end
.endc
.end