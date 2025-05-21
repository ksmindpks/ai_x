// 가변인자함수: 매개변수의 갯수에 따라 변하는 함수. 단, 화살표 함수에서는 불가
// 내장함수 Array()
var arr1 = [1, 2, '삼',];
var arr2 = Array(1, 2, '삼');
var arr3 = [, ,]; // 방의 갯수가 2인 배열
var arr4 = Array(2);
var arr5 = []; // 방의 갯수가 0인 배열
var arr6 = Array();
console.log(arr1, arr2, arr3, arr4, arr5, arr6);