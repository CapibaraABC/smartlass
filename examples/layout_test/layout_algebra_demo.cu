#include <iostream>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/util/print.hpp>
#include <cute/int_tuple.hpp>

#define PRINT 0

using namespace cute;

// test case
int g_test_total = 0;
int g_test_passed = 0;

// general ASSERT_EQ
template<typename T1, typename T2>
void ASSERT_EQ(const T1& val1, const T2& val2, const std::string& message = "") {
    ++g_test_total;
    if (val1 == val2) {
        ++g_test_passed;
        std::cout << "[PASS] " << message << "\n";
    } else {
        std::cout << "[FAIL] " << message << "\n";
        std::cout << "  Expected: "; print(val2); std::cout << "\n";
        std::cout << "  Actual  : "; print(val1); std::cout << "\n";
    }
}


/**
 * coalesce
 */
void test_1() {
    Layout A = make_layout(
        make_shape(_2{}, make_shape(_1{}, _6{})),
        make_stride(_1{}, make_stride(_6{}, _2{}))
    );

    Layout coalA = coalesce(A);
    Layout golden_coalA = Layout<_12, _1>{};
    ASSERT_EQ(coalA, golden_coalA, "coalesce Layout test1");

    Layout coalAByMode = coalesce(A, Step<_1, _1>{});
    Layout golden_coalAByMode = Layout<Shape<_2, _6>, Stride<_1, _2>>{};
    ASSERT_EQ(coalAByMode, golden_coalAByMode, "coalesce Layout test2");
#if PRINT
    print("A:");print(A);print("\n");
    print("coalA:");print(coalA);print("\n");
    print("coalAByMode:");print(coalAByMode);print("\n");
#endif
}


/**
 * composition
 */
void test_2() {
    Layout A = make_layout(
        make_shape(_10{}, _2{}),
        make_stride(_16{}, _4{})
    );

    Layout B = make_layout(
        make_shape(_5{}, _4{}),
        make_stride(_1{}, _5{})
    );

    Layout compoAB = composition(A, B);
    Layout golden_compoAB = Layout<Shape<_5, Shape<_2, _2>>, Stride<_16, Stride<_80, _4>>>{};
    ASSERT_EQ(compoAB, golden_compoAB, "composition Layout test1");

    //composition by mode
    Layout C = make_layout(
        make_shape(Int<12>{}, make_shape(Int<4>{}, Int<8>{})),
        make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{}))
    );
    auto tiler = make_tile(Layout<_3, _4>{}, Layout<_8, _2>{});
    Layout compoCTiler = composition(C, tiler);
    Layout golden_compoCTiler = Layout<Shape<Int<3>, Shape<Int<2>, Int<4>>>, Stride<Int<236>, Stride<Int<26>, Int<1>>>>{};
    ASSERT_EQ(compoCTiler, golden_compoCTiler, "composition Layout test2");

    Layout compoCTilerOther = make_layout(
        composition(layout<0>(C), get<0>(tiler)),
        composition(layout<1>(C), get<1>(tiler))
    );
    ASSERT_EQ(compoCTiler, compoCTilerOther, "composition Layout test3");       //cross itentify

    auto tiler2 = make_shape(_3{}, _8{});       //(3:1, 8:1)
    Layout compoCTiler2 = composition(C, tiler2);
    Layout golden_compoCTiler2 = Layout<Shape<Int<3>, Shape<Int<4>, Int<2>>>, Stride<Int<59>, Stride<Int<13>, Int<1>>>>{};
    ASSERT_EQ(compoCTiler2, golden_compoCTiler2, "composition Layout test4");

#if PRINT
    print("A:");print(A);print("\n");
    print("B:");print(B);print("\n");

    print("compoAB:");print(compoAB);print("\n");
    print("tiler:");print(tiler);print("\n");
    print("compoCTiler:");print(compoCTiler);print("\n");
    print("compoTilerOther:");print(compoCTilerOther);print("\n");

    print("tiler:");print(tiler);print("\n");
    print("compoCTiler2:");print(compoCTiler2);print("\n");
#endif
}


/**
 * complement
 */
void test_3() {
    Layout A = Layout<_4, _1>{};
    Layout compleA = complement(A, _24{});
    Layout golden_compleA = Layout<_6, _4>{};
    ASSERT_EQ(compleA, golden_compleA, "complement Layout test1");

    Layout B = Layout<_4, _2>{};
    Layout compleB = complement(B, _24{});
    Layout golden_compleB = Layout<Shape<_2, _3>, Stride<_1, _8>>{};
    ASSERT_EQ(compleB, golden_compleB, "complement Layout test2");

#if PRINT
    print("A:");print(A);print("\n");
    print("compleA:");print(compleA);print("\n");

    print("B:");print(B);print("\n");
    print("compleB:");print(compleB);print("\n");
#endif
}


/**
 * logical_divide:
 * ((TileM, RestM),(TileN, RestN),L,...)
 */
void test_4() {
    Layout A = make_layout(
        make_shape(_4{}, _2{}, _3{}),
        make_stride(_2{}, _1{}, _8{})
    );

    auto tiler = make_tile(Layout<_4, _2>{});
    Layout ldivA = logical_divide(A, tiler);
    Layout golden_ldivA = Layout<Shape<Shape<_4, _2>, _2, _3>, Stride<Stride<_4, _2>, _1, _8>>{};
    ASSERT_EQ(ldivA, golden_ldivA, "logical_divide Layout test1");


    /////logical_divide 2D//////
    Layout B = make_layout(
        make_shape(Int<9>{}, make_shape(Int<4>{}, Int<8>{})),
        make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{}))
    );
    auto tiler2d = make_tile(Layout<_3, _3>{}, Layout<Shape<_2, _4>, Stride<_1, _8>>{});
    Layout ldivB = logical_divide(B, tiler2d);
    Layout golden_ldivB = Layout<Shape<Shape<_3, _3>, Shape<Shape<_2, _4>, Shape<_2, _2>>>,
                                 Stride<Stride<Int<177>, Int<59>>, Stride<Stride<Int<13>, Int<2>>, Stride<Int<26>, Int<1>>>>
    >{};
    ASSERT_EQ(ldivB, golden_ldivB, "logical_divide Layout test2");

#if PRINT
    print("A:");print(A);print("\n");
    print("tiler:");print(tiler);print("\n");
    print("ldivA:");print(ldivA);print("\n");

    print("B:");print(B);print("\n");
    print("tiler2d:");print(tiler2d);print("\n");
    print("ldivB:");print(ldivB);print("\n");
#endif
}

/**
 * zipped_divide:
 * ((TileM, TileN),(RestM, RestN, L, ...))
 */
void test_5() {
    Layout A = make_layout(
        make_shape(_4{}, _2{}, _3{}),
        make_stride(_2{}, _1{}, _8{})
    );

    auto tiler = make_tile(Layout<_4, _2>{});
    Layout zdivA = zipped_divide(A, tiler);
    Layout golden_zdivA = Layout<Shape<Shape<_4>, Shape<_2, _2, _3>>, Stride<Stride<_4>, Stride<_2, _1, _8>>>{};
    ASSERT_EQ(zdivA, golden_zdivA, "zipped_divide Layout test1");


    //////////zipped_divide 2d//////////
    Layout B = make_layout(
        make_shape(Int<9>{}, make_shape(Int<4>{}, Int<8>{})),
        make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{}))
    );
    auto tiler2d = make_tile(Layout<_3, _3>{}, Layout<Shape<_2, _4>, Stride<_1, _8>>{});
    Layout zdivB = zipped_divide(B, tiler2d);
    Layout golden_zdivB = Layout<Shape<Shape<_3, Shape<_2, _4>>, Shape<_3, Shape<_2, _2>>>,
                                 Stride<Stride<Int<177>, Stride<Int<13>, Int<2>>>, Stride<Int<59>, Stride<Int<26>, Int<1>>>>
    >{};
    ASSERT_EQ(zdivB, golden_zdivB, "zipped_divide Layout test2");

#if PRINT
    print("A:");print(A);print("\n");
    print("tiler:");print(tiler);print("\n");
    print("zdivA:");print(zdivA);print("\n");

    print("B:");print(B);print("\n");
    print("tiler2d:");print(tiler2d);print("\n");
    print("zdivB:");print(zdivB);print("\n");
#endif
}

/**
 * tiled_divide:
 * ((TileM, TileN), RestM, RestN, L, ...)
 */
void test_6() {
    Layout A = make_layout(
        make_shape(_4{}, _2{}, _3{}),
        make_stride(_2{}, _1{}, _8{})
    );

    auto tiler = make_tile(Layout<_4, _2>{});
    Layout tdivA = tiled_divide(A, tiler);
    Layout golden_tdivA = Layout<Shape<Shape<_4>, _2, _2, _3>, Stride<Stride<_4>, _2, _1, _8>>{};
    ASSERT_EQ(tdivA, golden_tdivA, "tiled_divide Layout test1");


    //////////tiled_divide 2d//////////
    Layout B = make_layout(
        make_shape(Int<9>{}, make_shape(Int<4>{}, Int<8>{})),
        make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{}))
    );
    auto tiler2d = make_tile(Layout<_3, _3>{}, Layout<Shape<_2, _4>, Stride<_1, _8>>{});
    Layout tdivB = tiled_divide(B, tiler2d);
    Layout golden_tdivB = Layout<Shape<Shape<_3, Shape<_2, _4>>, _3, Shape<_2, _2>>,
                                 Stride<Stride<Int<177>, Stride<Int<13>, Int<2>>>, Int<59>, Stride<Int<26>, Int<1>>>
    >{};
    ASSERT_EQ(tdivB, golden_tdivB, "tiled_divide Layout test2");

#if PRINT
    print("A:");print(A);print("\n");
    print("tiler:");print(tiler);print("\n");
    print("tdivA:");print(tdivA);print("\n");
    print("golden_tdivA:");print(golden_tdivA);print("\n");

    print("B:");print(B);print("\n");
    print("tiler2d:");print(tiler2d);print("\n");
    print("tdivB:");print(tdivB);print("\n");
    print("golden_tdivB:");print(golden_tdivB);print("\n");
#endif
}

/**
 * flat_divide:
 * (TileM, TileN, RestM, RestN, L, ...)
 */
void test_7() {
    Layout A = make_layout(
        make_shape(_4{}, _2{}, _3{}),
        make_stride(_2{}, _1{}, _8{})
    );

    auto tiler = make_tile(Layout<_4, _2>{});
    Layout fdivA = flat_divide(A, tiler);
    Layout golden_fdivA = Layout<Shape<Shape<_4>, _2, _2, _3>, Stride<Stride<_4>, _2, _1, _8>>{};
    ASSERT_EQ(fdivA, golden_fdivA, "flat_divide Layout test1");


    //////////flat_divide 2d//////////
    Layout B = make_layout(
        make_shape(Int<9>{}, make_shape(Int<4>{}, Int<8>{})),
        make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{}))
    );
    auto tiler2d = make_tile(Layout<_3, _3>{}, Layout<Shape<_2, _4>, Stride<_1, _8>>{});
    Layout fdivB = flat_divide(B, tiler2d);
    Layout golden_fdivB = Layout<Shape<_3, Shape<_2, _4>, _3, Shape<_2, _2>>,
                                 Stride<Int<177>, Stride<Int<13>, Int<2>>, Int<59>, Stride<Int<26>, Int<1>>>
    >{};
    ASSERT_EQ(fdivB, golden_fdivB, "flat_divide Layout test2");

#if PRINT
    print("A:");print(A);print("\n");
    print("tiler:");print(tiler);print("\n");
    print("fdivA:");print(fdivA);print("\n");
    print("golden_fdivA:");print(golden_fdivA);print("\n");

    print("B:");print(B);print("\n");
    print("tiler2d:");print(tiler2d);print("\n");
    print("fdivB:");print(fdivB);print("\n");
    print("golden_fdivB:");print(golden_fdivB);print("\n");
#endif
}

/**
 * logical_product:
 * ((M, TileM),(N, TileN), L, ...)
 */
void test_8() {
    Layout A = make_layout(
        make_shape(_2{}, _2{}),
        make_stride(_4{}, _1{})
    );
    Layout tiler = Layout<_6, _1>{};
    Layout lproductA = logical_product(A, tiler);
    Layout golden_lproductA = Layout<Shape<Shape<_2, _2>, Shape<_2, _3>>, Stride<Stride<_4, _1>, Stride<_2, _8>>>{};
    ASSERT_EQ(lproductA, golden_lproductA, "logical_product Layout test1");

    //////////logical_product 2d//////////
    Layout B = make_layout(
        make_shape(_2{}, _5{}),
        make_stride(_5{}, _1{})
    );
    auto tiler2d = make_tile(Layout<_3, _5>{}, Layout<_4, _6>{});
    Layout lproductB = logical_product(B, tiler2d);
    Layout golden_lproductB = Layout<Shape<Shape<Int<2>, Int<3>>, Shape<Int<5>,Int<4>>>, 
                                    Stride<Stride<Int<5>, Int<10>>, Stride<Int<1>, Int<30>>>>{};
    ASSERT_EQ(lproductB, golden_lproductB, "logical_product Layout test2");
#if PRINT
    print("A:");print(A);print("\n");
    print("tiler:");print(tiler);print("\n");
    print("lproductA:");print(lproductA);print("\n");

    print("B:");print(B);print("\n");
    print("tiler2d:");print(tiler2d);print("\n");
    print("lproductB:");print(lproductB);print("\n");

#endif
}

/**
 * blocked_product: only for 1d
 * raked_product: only for 1d
 */
void test_9() {
    Layout A = make_layout(
        make_shape(_2{}, _2{}),
        make_stride(_4{}, _1{})
    );
    Layout tiler = Layout<_6, _1>{};
    Layout bproductA = blocked_product(A, tiler);
    Layout golden_bproductA = Layout<Shape<Shape<_2, Shape<_2, _3>>, Shape<_2, _1>>, 
                                    Stride<Stride<_4, Stride<_2, _8>>, Stride<_1, _0>>>{};
    ASSERT_EQ(bproductA, golden_bproductA, "blocked_product Layout test1");

    Layout rproductA = raked_product(A, tiler);
    Layout golden_rproductA = Layout<Shape<Shape<Shape<_2, _3>, _2>, Shape<_1, _2>>, 
                                            Stride<Stride<Stride<_2, _8>, _4>, Stride<_0, _1>>>{};
    ASSERT_EQ(rproductA, golden_rproductA, "raked_product Layout test1");
    
#if PRINT
    print("A:");print(A);print("\n");
    print("tiler:");print(tiler);print("\n");
    print("bproductA:");print(bproductA);print("\n");

    print("rproductA:");print(rproductA);print("\n");

#endif
}

/**
 * zipped_product:
 * ((M, N), (TileM, TileN, L, ...))
 */
void test_10() {
    Layout A = make_layout(
        make_shape(_2{}, _2{}),
        make_stride(_4{}, _1{})
    );
    Layout tiler = Layout<_6, _1>{};
    Layout zproductA = zipped_product(A, tiler);
    Layout golden_zproductA = Layout<Shape<Shape<_2, _2>, Shape<_2, _3>>, Stride<Stride<_4, _1>, Stride<_2, _8>>>{};
    ASSERT_EQ(zproductA, golden_zproductA, "zipped_product Layout test1");

    //////////zipped_product 2d//////////
    Layout B = make_layout(
        make_shape(_2{}, _5{}),
        make_stride(_5{}, _1{})
    );
    auto tiler2d = make_tile(Layout<_3, _5>{}, Layout<_4, _6>{});
    Layout zproductB = zipped_product(B, tiler2d);
    Layout golden_zproductB = Layout<Shape<Shape<Int<2>, Int<5>>, Shape<Int<3>,Int<4>>>, 
                                    Stride<Stride<Int<5>, Int<1>>, Stride<Int<10>, Int<30>>>>{};
    ASSERT_EQ(zproductB, golden_zproductB, "zipped_product Layout test2");
#if PRINT
    print("A:");print(A);print("\n");
    print("tiler:");print(tiler);print("\n");
    print("zproductA:");print(zproductA);print("\n");

    print("B:");print(B);print("\n");
    print("tiler2d:");print(tiler2d);print("\n");
    print("zproductB:");print(zproductB);print("\n");

#endif
}

/**
 * tiled_product:
 * ((M,N), TileM, TileN, L, ...)
 */
void test_11() {
    Layout A = make_layout(
        make_shape(_2{}, _2{}),
        make_stride(_4{}, _1{})
    );
    Layout tiler = Layout<_6, _1>{};
    Layout tproductA = tiled_product(A, tiler);
    Layout golden_tproductA = Layout<Shape<Shape<_2, _2>, _2, _3>, Stride<Stride<_4, _1>, _2, _8>>{};
    ASSERT_EQ(tproductA, golden_tproductA, "tiled_product Layout test1");

    //////////tiled_product 2d//////////
    Layout B = make_layout(
        make_shape(_2{}, _5{}),
        make_stride(_5{}, _1{})
    );
    auto tiler2d = make_tile(Layout<_3, _5>{}, Layout<_4, _6>{});
    Layout tproductB = tiled_product(B, tiler2d);
    Layout golden_tproductB = Layout<Shape<Shape<Int<2>, Int<5>>, Int<3>,Int<4>>, 
                                    Stride<Stride<Int<5>, Int<1>>, Int<10>, Int<30>>>{};
    ASSERT_EQ(tproductB, golden_tproductB, "tiled_product Layout test2");
#if PRINT
    print("A:");print(A);print("\n");
    print("tiler:");print(tiler);print("\n");
    print("tproductA:");print(tproductA);print("\n");

    print("B:");print(B);print("\n");
    print("tiler2d:");print(tiler2d);print("\n");
    print("tproductB:");print(tproductB);print("\n");

#endif
}

/**
 * flat_product:
 * (M, N, TileM, TileN, L, ...)
 */
void test_12() {
    Layout A = make_layout(
        make_shape(_2{}, _2{}),
        make_stride(_4{}, _1{})
    );
    Layout tiler = Layout<_6, _1>{};
    Layout fproductA = flat_product(A, tiler);
    Layout golden_fproductA = Layout<Shape<_2, _2, _2, _3>, Stride<_4, _1, _2, _8>>{};
    ASSERT_EQ(fproductA, golden_fproductA, "flat_product Layout test1");

    //////////flat_product 2d//////////
    Layout B = make_layout(
        make_shape(_2{}, _5{}),
        make_stride(_5{}, _1{})
    );
    auto tiler2d = make_tile(Layout<_3, _5>{}, Layout<_4, _6>{});
    Layout fproductB = flat_product(B, tiler2d);
    Layout golden_fproductB = Layout<Shape<Int<2>, Int<5>, Int<3>,Int<4>>, 
                                    Stride<Int<5>, Int<1>, Int<10>, Int<30>>>{};
    ASSERT_EQ(fproductB, golden_fproductB, "flat_product Layout test2");
#if PRINT
    print("A:");print(A);print("\n");
    print("tiler:");print(tiler);print("\n");
    print("fproductA:");print(fproductA);print("\n");

    print("B:");print(B);print("\n");
    print("tiler2d:");print(tiler2d);print("\n");
    print("fproductB:");print(fproductB);print("\n");

#endif
}

int main(){
    test_1();       //coalesce
    test_2();       //composition
    test_3();       //complement
    test_4();       //logical_divide
    test_5();       //zipped_divide
    test_6();       //tiled_divide
    test_7();       //flat_divide
    test_8();       //logical_product
    test_9();       //blocked_product, raked_product
    test_10();      //zipped_product
    test_11();      //tiled_product
    test_12();      //flat_product

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed " << g_test_passed << "/" << g_test_total << " tests.\n";
    return 0;
}