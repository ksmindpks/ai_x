let funcVar = function(){
    console.log('1. 매개변수 없는 일반함수 호출');
}
funcVar();

funcVar = () => {
    console.log('2. 매개변수 없는 화살표 함수 호출');
}
funcVar();

funcVar =function(i){
    console.log('3. 매개변수 하나있는 일반 함수 호출');
    console.log('매개변수 값=',i);
}
funcVar(10);
funcVar = i => {
    console.log('4. 매개변수 하나있는 화살표 함수 호출');
    console.log('매개변수 값=',i);
}
funcVar(10);
funcVar = function(i) {
    console.log('5. 매개변수 하나의 한줄짜리 일반 함수 호출',i);
}
funcVar(20);
funcVar = i => console.log('6. 매개변수 하나의 한줄짜리 화살표 함수 호출',i);
funcVar(20);

funcVar = function(x){
    return x*x;
}
console.log('7. return문만 있는 일반반 함수 호출',funcVar(10));
funcVar = x => x*x;
console.log('8. return문만 있는 화살표 함수 호출', funcVar(7))

funcVar = function(x,y){
    return 10*x+y;
}
funcVar = (x,y) => 10*x+y;
console.log('9. 매개변수 2개짜리 return 문장의 화살표 함수 호출', funcVar(5,3))


var arr=[10, '홍길동', '신림동'];
arr.forEach(function(data, idx){
    console.log(idx, '번째 :', data)
});

arr.forEach((data, idx) => console.log(idx, '번째 :', data))

arr.forEach(function(data){
    console.log(data);
});
arr.forEach(data => console.log(data));
