/* array함수 : 가변인자함수(파이썬에선 튜플매개변수)
*  매개변수 0개: length가 0인 배열 생성 return
*  매개변수 1개: lenght가 매개변수만큼의 크기인 배열 생성 return
*  매개변수 2개 이상: 매개변수로 배열을 생성 return
*/

function array(){ //
    // console.log(arguments);
    // console.log(arguments.length);
    let rslt = [];
    if(arguments.length==1){
        // rslt를 arguments[0]만큼의 크기인 배열
        // arguments.forEach는 사용불가가
        for(let cnt=1; cnt<arguments[0]; cnt++){
            rslt.push(null);
        }
    }else if(arguments.length>=2){
        // arguments의 내용으로 배열
        for(var idx=0; idx<arguments.length; idx++){
            rslt.push(arguments[idx]);
        }
    }
    return rslt;
}
var arr1 = array();
var arr2 = array(3);
// var arr = array(2,3);
var arr3 = array(2,3,'사');
console.log(arr1, arr2, arr3);