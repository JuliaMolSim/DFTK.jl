@testitem "Phonon: Ewald: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, TestCases] begin
    using DFTK

    testcase = TestCases.silicon
    terms = [Ewald()]
    ω_ref = [ -0.2442311083805831
              -0.24423110838058237
              -0.23442208238107232
              -0.23442208238107184
              -0.1322944535508822
              -0.13229445355088176
              -0.10658869539441493
              -0.10658869539441468
              -0.10658869539441346
              -0.10658869539441335
              -4.891274318712944e-16
              -3.773447798738169e-17
              1.659776058962626e-15
              0.09553958285993536
              0.18062696253387409
              0.18062696253387464
              0.4959725605665635
              0.4959725605665648
              0.49597256056656597
              0.5498259359834827
              0.5498259359834833
              0.6536453595829087
              0.6536453595829091
              0.6536453595829103
              0.6536453595829105
              0.6961890494198791
              0.6961890494198807
              0.7251587593311752
              0.7251587593311782
              0.9261195383192374
              0.9261195383192381
              1.2533843205271504
              1.2533843205271538
              1.7010950724885228
              1.7010950724885254
              1.752506588801463]
    Phonon.test_frequencies(testcase, terms, ω_ref)
end

@testitem "Phonon: Ewald: comparison to automatic differentiation" #=
    =#    tags=[:phonon, :slow, :dont_test_mpi] setup=[Phonon, TestCases] begin
    using DFTK
    testcase = TestCases.silicon

    terms = [Ewald()]
    Phonon.test_rand_frequencies(testcase, terms)
end
