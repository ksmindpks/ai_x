console.log(pow(9,3));
// 선언된 매개변수보다 많은 매개변수로 호출 경우 뒷부분은 무시시
console.log(pow(2,3,4));
// 선언된 매개변수보다 적은 매개변수로 호출 경우 전달되지 않은 파라미터는 undefined로
console.log(pow(5));
console.log(pow());
function pow(x, y){
    // x의 y승을 return
    console.log('함수내의 x=', x, ' / y=', y);
    let rslt = 1;
    for(let cnt=1; cnt<=y; cnt++){
        rslt *= x; // rslt=rslt*x;
    }
    // return rslt; // return이 없으면 undefined로 받음
}