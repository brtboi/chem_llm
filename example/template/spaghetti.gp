# set Fermi energy to correct value
Efermi=1.8976
# ... and uncomment the following line
set terminal pdf
set output 'bands.pdf'
set xzeroaxis lt -1

set grid xtics lt -1 lw 1
set tics font ", 18"
set format y "%5.1f"
set format x ""
set ylabel "Energy (Ry)" font ", 18"
set yrange [-5.0:6.0]
unset xlabel

set xtics ("R" 0.0, "{/Symbol G}" 0.8016, "X" 1.3016, "M" 1.8179, "{/Symbol G}" 2.5366)

plot [0:2.5366] 'CsPbI3.1.0.bands.dat.gnu' u 1:($2-Efermi) notitle w lines lw 1.5 lt rgb "blue" #points lw 3 pt 6 

unset output
