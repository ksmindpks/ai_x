// A가 만든 부분


function pythagoras(width, height){
    // 내부함수 : 함수 내의 함수
    // 내부함수  사용 이유 : 충돌을 피하고자
    // function square(x){
    //     return x*x;
    // }
    const square = x => x*x;
    return Math.sqrt(square(width)+square(height));
}