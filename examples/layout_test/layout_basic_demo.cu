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

////////////basic API for IntTuple////////////

/**
 * IntTuple:
 * make_tuple(int{2}, Int<3>{});
 */
void test_1() {
    auto make_tuple1 = make_tuple(int{2}, Int<3>{});
    auto golden1 = make_tuple(2, _3{});

    ASSERT_EQ(make_tuple1, golden1, "Tuple equality test");
}

/**
 * get<I>IntTuple
 * 
 */
void test_2() {
    auto tuple1 = make_tuple(int{2}, Int<3>{});
    auto get1 = get<0>(tuple1);
    auto golden_get1 = 2;
    ASSERT_EQ(get1, golden_get1, "get Tuple test1");

    auto tuple2 = make_tuple(make_tuple(_1{}, _2{}), 
                             make_tuple(_3{}, _4{}, make_tuple(_5{}, _6{})),
                             make_tuple(_5{}, _6{})
                            );
    auto get2 = get<0>(tuple2);
    auto golden_get2 = make_tuple(_1{}, _2{});
    ASSERT_EQ(get2, golden_get2, "get Tuple test2");

    auto get3 = get<1>(tuple2);
    auto golden_get3 = make_tuple(_3{}, _4{}, make_tuple(_5{}, _6{}));
    ASSERT_EQ(get3, golden_get3, "get Tuple test3");
}

/**
 * depth<I>IntTuple
 * 
 */
void test_3() {
    auto tuple1 = make_tuple(int{2}, Int<3>{});
    auto dep = depth(tuple1);
    auto golden_dep = _1{};
    ASSERT_EQ(dep, golden_dep, "depth Tuple test1");

    auto tuple2 = make_tuple(make_tuple(_1{}, _2{}), 
                             make_tuple(_3{}, _4{}, make_tuple(_5{}, _6{})),
                             make_tuple(_5{}, _6{})
                            );
    auto dep0 = depth<0>(tuple2);
    auto golden_dep0 = _1{};
    ASSERT_EQ(dep0, golden_dep0, "depth Tuple test2");

    auto dep1 = depth<1>(tuple2);
    auto golden_dep1 = _2{};
    ASSERT_EQ(dep1, golden_dep1, "depth Tuple test3");

    auto dep00 = depth<0,0>(tuple2);
    auto golden_dep00 = _0{};
    ASSERT_EQ(dep00, golden_dep00, "depth Tuple test4");

    auto dep12 = depth<1,2>(tuple2);
    auto golden_dep12 = _1{};
    ASSERT_EQ(dep12, golden_dep12, "depth Tuple test5");

    auto dep_ = depth(tuple2);
    auto golden_dep_ = _3{};
    ASSERT_EQ(dep_, golden_dep_, "depth Tuple test6");

#if PRINT
    print("tuple1:");print(tuple1);print("\n");
    print("dep:");print(dep);print("\n");

    print("tuple2:");print(tuple2);print("\n");
    print("dep0:");print(dep0);print("\n");

    print("dep1:");print(dep1);print("\n");
    print("dep00:");print(dep00);print("\n");
    print("dep12:");print(dep12);print("\n");
    print("dep_:");print(dep_);print("\n");
    
#endif
}

/**
 * rank<>(Tuple)
 * return the elements in Tuple, and elements may own diffrent types. 
 */
void test_4() {
    auto tuple = make_tuple(make_tuple(_1{}, _2{}), 
                             make_tuple(_3{}, _4{}, make_tuple(_5{}, _6{})),
                             make_tuple(_5{}, _6{})
                            );
    auto rank_ = rank(tuple);
    auto golden_rank_ = _3{};
    ASSERT_EQ(rank_, golden_rank_, "rank Tuple test1");

    auto rank0 = rank<0>(tuple);
    auto golden_rank0 = _2{};
    ASSERT_EQ(rank0, golden_rank0, "rank Tuple test2");

    auto rank1 = rank<1>(tuple);
    auto golden_rank1 = _3{};
    ASSERT_EQ(rank1, golden_rank1, "rank Tuple test3");

    auto rank12 = rank<1,2>(tuple);
    auto golden_rank12 = _2{};
    ASSERT_EQ(rank12, golden_rank12, "rank Tuple test4");

    auto rank120 = rank<1,2,0>(tuple);
    auto golden_rank120 = _1{};
    ASSERT_EQ(rank120, golden_rank120, "rank Tuple test5");

#if PRINT
    print("tuple:");print(tuple);print("\n");
    print("rank_:");print(rank_);print("\n");

    print("rank0:");print(rank0);print("\n");
    print("rank1:");print(rank1);print("\n");
    print("rank12:");print(rank12);print("\n");
    print("rank120:");print(rank120);print("\n");
    
#endif

}

////////////basic API for Layout////////////

/**
 * make_layout
 */
void test_5() {
    Layout layoutA = make_layout(
        make_shape(_1{}, _2{}),
        make_stride(_0{}, _3{})
    );
    Layout golden_layoutA = Layout<Shape<_1, _2>, Stride<_0,_3>>{};
    ASSERT_EQ(layoutA, golden_layoutA, "make Layout test1");

    Layout layoutB = make_layout(
        make_shape(make_shape(_1{}, _2{}), make_shape(_3{}, _4{})),
        make_stride(make_stride(_0{}, _2{}), make_stride(_1{}, _2{}))
    );
    Layout golden_layoutB = Layout<Shape<Shape<_1, _2>, Shape<_3, _4>>, 
                                  Stride<Stride<_0, _2>, Stride<_1, _2>>>{};
    ASSERT_EQ(layoutB, golden_layoutB, "make Layout test2");

    //col-major
    Layout layoutColMajor = make_layout(
        make_shape(make_shape(_10{}, _4{}), _5{}),
        LayoutLeft{}
    );
    auto golden_layoutColMajor = Layout<Shape<Shape<_10, _4>, _5>, Stride<Stride<_1, _10>, _40>>{};
    ASSERT_EQ(layoutColMajor, golden_layoutColMajor, "make Layout test3");

    //row-major
    Layout layoutRowMajor = make_layout(
        make_shape(make_shape(_10{}, _4{}), _5{}),
        LayoutRight{}
    );
    auto golden_layoutRowMajor = Layout<Shape<Shape<_10, _4>, _5>, Stride<Stride<Int<20>, _5>, _1>>{};
    ASSERT_EQ(layoutRowMajor, golden_layoutRowMajor, "make Layout test4");

#if PRINT
    print("layoutA:");print(layoutA);print("\n");
    print("golden_layoutA:");print(golden_layoutA);print("\n");

    print("layoutB:");print(layoutB);print("\n");
    print("golden_layoutB:");print(golden_layoutB);print("\n");

    print("layoutColMajor:");print(layoutColMajor);print("\n");
    print("layoutRowMajor:");print(layoutRowMajor);print("\n");
#endif
}


/***
 * get<>(Layout)
 */
void test_6() {
    Layout layoutA = make_layout(
        make_shape(make_shape(_1{}, _2{}), make_shape(_3{}, _4{})),
        make_stride(make_stride(_0{}, _2{}), make_stride(_1{}, _2{}))
    );
    auto get0 = get<0>(layoutA);
    Layout golden_get0 = Layout<Shape<_1, _2>, Stride<_0,_2>>{};
    ASSERT_EQ(get0, golden_get0, "get Layout test1");

    auto get01 = get<0,0>(layoutA);
    Layout golden_get01 = Layout<_1>{};
    ASSERT_EQ(get01, golden_get01, "get Layout test2");

    Layout layoutB = make_layout(get<1>(layoutA), get<0>(layoutA));
    Layout golden_layoutB = Layout<Shape<Shape<_3, _4>, Shape<_1, _2>>, 
                                   Stride<Stride<_1, _2>, Stride<_0, _2>>>{};
    ASSERT_EQ(layoutB, golden_layoutB, "get Layout test3");

    
#if PRINT
    print("layoutA:");print(layoutA);print("\n");
    print("get0:");print(get0);print("\n");
    print("golden_get0:");print(golden_get0);print("\n");
    print("get01:");print(get01);print("\n");
    print("golden_get01:");print(golden_get01);print("\n");

    print("layoutB:");print(layoutB);print("\n");

#endif
}

/**
 * rank<>(Layout)
 */
void test_7(){
    Layout layoutA = make_layout(
        make_shape(make_shape(_1{}, _2{}), make_shape(_3{}, _4{})),
        make_stride(make_stride(_0{}, _2{}), make_stride(_1{}, _2{}))
    );
    auto rankA = rank(layoutA);
    auto golden_rankA = _2{};
    ASSERT_EQ(rankA, golden_rankA, "rank Layout test1");

    auto rank0 = rank<0>(layoutA);
    auto golden_rank0 = _2{};
    ASSERT_EQ(rank0, golden_rank0, "rank Layout test2");

    auto rank00 = rank<0,0>(layoutA);
    auto golden_rank00 = _1{};
    ASSERT_EQ(rank00, golden_rank00, "rank Layout test3");

    
#if PRINT
    print("layoutA:");print(layoutA);print("\n");
    print("rankA:");print(rankA);print("\n");
    print("rank0:");print(rank0);print("\n");

    print("rank00:");print(rank00);print("\n");
#endif
}

/**
 * depth of Layout
 * depth<>(Layout)
 */
void test_8() {
    Layout layoutA = make_layout(
        make_shape(make_shape(_1{}, _2{}), make_shape(_3{}, _4{})),
        make_stride(make_stride(_0{}, _2{}), make_stride(_1{}, _2{}))
    );

    auto depA = depth(layoutA);
    auto golden_depA = _2{};
    ASSERT_EQ(depA, golden_depA, "depth Layout test1");

    auto dep0 = depth<0>(layoutA);
    auto golden_dep0 = _1{};
    ASSERT_EQ(dep0, golden_dep0, "depth Layout test2");

    auto dep00 = depth<0,0>(layoutA);
    auto golden_dep00 = _0{};
    ASSERT_EQ(dep00, golden_dep00, "depth Layout test3");

#if PRINT
    print("layoutA:");print(layoutA);print("\n");
    print("depA:");print(depA);print("\n");
    print("dep0:");print(dep0);print("\n");

    print("dep00:");print(dep00);print("\n");
#endif
}

/**
 * shape, stride
 */
void test_9() {
    Layout layoutA = make_layout(
        make_shape(make_shape(_1{}, _2{}), make_shape(_3{}, _4{})),
        make_stride(make_stride(_0{}, _2{}), make_stride(_1{}, _2{}))
    );
    //////////shape//////////
    auto shapeA = shape(layoutA);
    auto golden_shapeA = Shape<Shape<_1, _2>, Shape<_3, _4>>{};
    ASSERT_EQ(shapeA, golden_shapeA, "shape Layout test1");

    auto shape0 = shape<0>(layoutA);
    auto golden_shape0 = Shape<_1, _2>{};
    ASSERT_EQ(shape0, golden_shape0, "shape Layout test2");

    auto shape00 = shape<0,0>(layoutA);
    auto golden_shape00 = _1{};
    ASSERT_EQ(shape00, golden_shape00, "shape Layout test3");

    //////////stride//////////
    auto strideA = stride(layoutA);
    auto golden_strideA = Stride<Stride<_0, _2>, Stride<_1, _2>>{};
    ASSERT_EQ(strideA, golden_strideA, "stride Layout test1");

    auto stride0 = stride<0>(layoutA);
    auto golden_stride0 = Stride<_0, _2>{};
    ASSERT_EQ(stride0, golden_stride0, "stride Layout test2");

    auto stride01 = stride<0,1>(layoutA);
    auto golden_stride01 = _2{};
    ASSERT_EQ(stride01, golden_stride01, "stride Layout test3");

#if PRINT
    print("layoutA:");print(layoutA);print("\n");
    print("////////shape:////////\n");
    print("shapeA:");print(shapeA);print("\n");
    print("shape0:");print(shape0);print("\n");
    print("shape00:");print(shape00);print("\n");

    print("////////stride:////////\n");
    print("strideA:");print(strideA);print("\n");
    print("stride0:");print(stride0);print("\n");
    print("stride01:");print(stride01);print("\n");
#endif
}

/**
 * size, cosize
 */
void test_10() {
    Layout layoutA = make_layout(
        make_shape(make_shape(_1{}, _2{}), make_shape(_3{}, _4{})),
        make_stride(make_stride(_0{}, _2{}), make_stride(_1{}, _2{}))
    );
    //////////size//////////
    auto sizeA = size(layoutA);
    auto golden_sizeA = _24{};
    ASSERT_EQ(sizeA, golden_sizeA, "size Layout test1");

    auto size0 = size<0>(layoutA);
    auto golden_size0 = _2{};
    ASSERT_EQ(size0, golden_size0, "size Layout test2");

    auto size11 = size<1,1>(layoutA);
    auto golden_size11 = _4{};
    ASSERT_EQ(size11, golden_size11, "size Layout test3");

    //////////cosize//////////
    auto cosizeA = cosize(layoutA);     //(0,1,2,3)*(0,2,1,2)+1 = 11
    auto golden_cosizeA = Int<11>{};
    ASSERT_EQ(cosizeA, golden_cosizeA, "cosize Layout test1");

    auto cosize0 = cosize<0>(layoutA);
    auto golden_cosize0 = Int<3>{};
    ASSERT_EQ(cosize0, golden_cosize0, "cosize Layout test2");

    auto cosize11 = cosize<1,1>(layoutA);       //3*2+1=7
    auto golden_cosize11 = Int<7>{};
    ASSERT_EQ(cosize11, golden_cosize11, "cosize Layout test3");
#if PRINT
    print("layoutA:");print(layoutA);print("\n");
    print("////////size:////////\n");
    print("sizeA:");print(sizeA);print("\n");
    print("size0:");print(size0);print("\n");
    print("size11:");print(size11);print("\n");

    print("////////cosize:////////\n");
    print("cosizeA:");print(cosizeA);print("\n");
    print("cosize0:");print(cosize0);print("\n");
    print("cosize11:");print(cosize11);print("\n");
#endif
}


/**
 * idx2crd, crd2idx
 */
void test_11() {

    auto shapeA = make_shape(make_shape(_1{}, _2{}), make_shape(_3{}, _4{}));  //((_1, _2), (_3, _4))

    auto idx0 = crd2idx(make_coord(make_coord(_0{}, _1{}), make_coord(_2{}, _3{})), shapeA);
    auto golden_idx0 = Int<23>{};
    ASSERT_EQ(idx0, golden_idx0, "crd2idx test1");

    auto crd0 = idx2crd(Int<9>{}, shapeA);
    auto golden_crd0 = make_coord(make_coord(_0{}, _1{}), make_coord(_1{}, _1{}));
    ASSERT_EQ(crd0, golden_crd0, "idx2crd test1");

    auto crd1 = idx2crd(Int<13>{}, shapeA);
    auto golden_crd1 = make_coord(make_coord(_0{}, _1{}), make_coord(_0{}, _2{}));
    ASSERT_EQ(crd1, golden_crd1, "idx2crd test2");

#if PRINT
    print("shapeA:");print(shapeA);print("\n");
    print("idx0:");print(idx0);print("\n");
    print("crd0:");print(crd0);print("\n");
    print("crd1:");print(crd1);print("\n");
#endif
}

/**
 * subLayout, select, take
 */
void test_12() {
    Layout layoutA = make_layout(
        make_shape(make_shape(_1{}, _2{}), make_shape(_3{}, _4{})),
        make_stride(make_stride(_0{}, _2{}), make_stride(_1{}, _2{}))
    );

    //////////subLayout//////////
    Layout layout0 = layout<0>(layoutA);        //(_1,_2):(_0,_2)
    Layout get0 = get<0>(layoutA);
    Layout golden_layout0 = Layout<Shape<_1, _2>, Stride<_0, _2>>{};
    ASSERT_EQ(layout0, golden_layout0, "subLayout test1");
    ASSERT_EQ(layout0, get0, "subLayout test2");            //cross identify

    Layout layout01 = layout<0,1>(layoutA);
    Layout get01 = get<0,1>(layoutA);
    Layout golden_layout01 = Layout<_2, _2>{};
    ASSERT_EQ(layout01, golden_layout01, "subLayout test3");
    ASSERT_EQ(layout01, get01, "subLayout test4");

    //////////select//////////
    Layout sel0 = select<0>(layoutA);                       //((_1,_2)):((_0,_2))
    Layout golden_sel0 = Layout<Shape<Shape<_1, _2>>, Stride<Stride<_0, _2>>>{};
    ASSERT_EQ(sel0, golden_sel0, "select Layout test1");

    Layout sel10 = select<1,0>(layoutA);
    Layout golden_sel10 = Layout<Shape<Shape<_3, _4>, Shape<_1, _2>>, Stride<Stride<_1, _2>, Stride<_0, _2>>>{};
    ASSERT_EQ(sel10, golden_sel10, "select Layout test2");

    Layout sel110 = select<1,1,0>(layoutA);
    Layout golden_sel110 = Layout<Shape<Shape<_3, _4>, Shape<_3, _4>, Shape<_1, _2>>, 
                                  Stride<Stride<_1, _2>, Stride<_1, _2>, Stride<_0, _2>>>{};
    ASSERT_EQ(sel110, golden_sel110, "select Layout test3");

    //////////take//////////
    Layout layoutB = make_layout(make_shape(_5{}, _2{}, _1{}, _4{}), LayoutRight{});
    Layout take01 = take<0,1>(layoutB); //(_5):(_8)
    // Layout getB0 = get<0>(layoutB);     //_5:_8
    Layout golden_take01 = Layout<Shape<_5>, Stride<_8>>{};
    ASSERT_EQ(take01, golden_take01, "take Layout test1");

    Layout take14 = take<1,4>(layoutB);
    Layout golden_take14 = Layout<Shape<_2, _1, _4>, Stride<_4, _0, _1>>{};
    ASSERT_EQ(take14, golden_take14, "take Layout test2");

#if PRINT
    print("layoutA:");print(layoutA);print("\n");
    print("layout0:");print(layout0);print("\n");
    print("get0:");print(get0);print("\n");
    print("golden_layout0:");print(golden_layout0);print("\n");

    print("layout01:");print(layout01);print("\n");
    print("get01:");print(get01);print("\n");
    print("golden_layout01:");print(golden_layout01);print("\n");

    print("sel0:");print(sel0);print("\n");
    print("sel10:");print(sel10);print("\n");
    print("sel110:");print(sel110);print("\n");

    print("layoutB:");print(layoutB);print("\n");
    print("take01:");print(take01);print("\n");

    print("take14:");print(take14);print("\n");
#endif
}

/**
 * concatenation, append, prepend, replace
 */
void test_13() {
    Layout A = make_layout(_3{}, _1{});     //_3:_1
    Layout B = make_layout(_4{}, _3{});     //_4:_3

    Layout row = make_layout(A,B);          //(_3,_4):(_1,_3)
    Layout col = make_layout(B,A);          //(_4,_3):(_3,_1)

    Layout compLayout = make_layout(row, col);
    Layout golden_compLayout = Layout<Shape<Shape<_3, _4>, Shape<_4, _3>>, 
                                      Stride<Stride<_1, _3>, Stride<_3, _1>>>{};
    ASSERT_EQ(compLayout, golden_compLayout, "concatenation Layout test1");

    Layout ab = append(A, B);
    Layout golden_ab = Layout<Shape<_3, _4>, Stride<_1, _3>>{};
    ASSERT_EQ(ab, golden_ab, "append Layout test1");

    Layout ba = append(B, A);
    Layout pre_ab = prepend(A, B);
    Layout golden_ba = Layout<Shape<_4, _3>, Stride<_3, _1>>{};
    ASSERT_EQ(ba, golden_ba, "prepend Layout test1");
    ASSERT_EQ(ba, pre_ab, "prepend Layout test2");          //cross itentify

    Layout c = append(ab, ab);                              //(_3,_4, (_3,_4)):(_1,_3,(_1,_3))
    Layout d = replace<0>(c, B);                               //(_4,_4, (_3,_4)):(_3,_3,(_1,_3))
    Layout golden_d = Layout<Shape<_4, _4, Shape<_3, _4>>, Stride<_3, _3, Stride<_1, _3>>>{};
    ASSERT_EQ(d, golden_d, "select Layout test1");

#if PRINT
    print("A:");print(A);print("\n");
    print("B:");print(B);print("\n");
    print("row:");print(row);print("\n");
    print("col:");print(col);print("\n");

    print("compLayout:");print(compLayout);print("\n");

    print("ab:");print(ab);print("\n");
    print("ba:");print(ba);print("\n");
    print("pre_ab:");print(pre_ab);print("\n");

    print("d:");print(d);print("\n");
#endif
}

/**
 * group<ModeBegin, ModeEnd>, 
 * flatten
 */
void test_14() {
    Layout A = make_layout(
        make_shape(_2{}, _4{}, _6{}, _1{}, _5{}),
        LayoutLeft{}
    );          //(_2, _4, _6, _1, _5):(_1, _2, _8, _0, _48)

    Layout group02 = group<0,2>(A);
    Layout golden_group02 = Layout<Shape<Shape<_2, _4>, _6, _1, _5>, 
                                   Stride<Stride<_1, _2>, _8, _0, Int<48>>>{};
    ASSERT_EQ(group02, golden_group02, "group Layout test1");

    Layout group24 = group<2,4>(group02);
    Layout golden_group24 = Layout<Shape<Shape<_2, _4>, _6, Shape<_1, _5>>, 
                                   Stride<Stride<_1, _2>, _8, Shape<_0, Int<48>>>>{};
    ASSERT_EQ(group24, golden_group24, "group Layout test2");

    Layout f02 = flatten(group02);
    Layout golden_f02 = Layout<Shape<_2, _4, _6, _1, _5>, Stride<_1, _2, _8, _0, Int<48>>>{};
    ASSERT_EQ(f02, golden_f02, "flatten Layout test1");

    Layout f24 = flatten(group24);
    ASSERT_EQ(f24, f02, "flatten Layout test2");            //cross itentify

#if PRINT
    print("A:");print(A);print("\n");
    print("group02:");print(group02);print("\n");
    print("group24:");print(group24);print("\n");

    print("f02:");print(f02);print("\n");
    print("f24:");print(f24);print("\n");
#endif
}

int main() {
    test_1();       //make Tuple
    test_2();       //get<>(Tuple)
    test_3();       //depth<>(Tuple)
    test_4();       //rank<>(Tuple)

    test_5();       //make Layout
    test_6();       //get<>(Layout)
    test_7();       //rank<>(Layout)
    test_8();       //depth<>(Layout)
    test_9();       //shape, stride
    test_10();      //size, cosize
    test_11();      //idx2crd, crd2idx
    test_12();      //sublayout, select, take
    test_13();      //concatenation, append, prepend, replace
    test_14();      //group, flatten

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed " << g_test_passed << "/" << g_test_total << " tests.\n";

    return 0;
}