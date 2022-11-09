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

let ls_start = 747p
let ls_test =747p
let ls_stop =754p
let delta_ls =1p

let ld_start = 3.7n
let ld_test =3.7n
let ld_stop =4.4n
let delta_ld =0.1n

let lg_start = 9.4n
let lg_test =9.4n
let lg_stop =10.8n
let delta_lg =0.2n

let w_start = 73u
let w_test =73u
let w_stop =76.5u
let delta_w =0.5u

while ls_test le ls_stop
    alter Ls = ls_test
    while ld_test le ld_stop
        alter Ld = ld_test
        while lg_test le lg_stop
            alter Lg = lg_test
                while w_test le w_stop
                    alter @mn2[w] = w_test
                     alter @mn1[w] = w_test
			sp lin 100 10M 10G 1
			let s21 = abs(S_2_1)
			let s12 = abs(S_1_2)
			let s11 = abs(S_1_1)
                  let s11db = 10*log(s11)
			let reals11 = minimum((S11db))

			let G1 = (real(s21))^2
			let G2 = vecmax(G1)
			print G2 >> Gt.csv


			print reals11 >> s11.csv
			let nf = real(minimum(NF))
			print nf >> nf.csv
	
                    

                let w_test = w_test + delta_w
                end
            let w_test = w_start

        let lg_test = lg_test + delta_lg
        end
let lg_test= lg_start

    let ld_test = ld_test + delta_ld
    end
let ld_test= ld_start

let ls_test = ls_test + delta_ls
end
	

          .endc
      .end