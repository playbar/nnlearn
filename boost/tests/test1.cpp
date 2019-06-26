#include <boost/test/included/unit_test.hpp>

#include <boost/test/parameterized_test.hpp>
#include <boost/thread.hpp>

using namespace boost::unit_test;


static void thread1_test()
{
	std::string strTmp = "thread1_test";
	for (int i = 0; i< 1000; i++) {
		printf("%d \n", i);
	}
}

void free_test_function( int i )
{
    BOOST_CHECK( i < 4 );
}


test_suite* init_unit_test_suite( int argc, char* argv[] )
{
    boost::thread t1(boost::bind(&thread1_test));

    int params[] = { 1, 2, 3, 4, 5 };
    framework::master_test_suite().
            add( BOOST_PARAM_TEST_CASE( &free_test_function, params, params+5 ) );

    return 0;

}