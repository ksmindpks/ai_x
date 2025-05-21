function sumAll(){
    let rslt=0;
    // 매개변수가 없으면 -999를 리턴
    if(arguments.length==0){
        rslt = -999
    }
    // 매개변수가 1개 이상이면 누적값리턴
    else{
        for(var data of arguments){
            rslt += data;
        }
    }
    return rslt;
}
// test
// console.log(sumAll());
// console.log(sumAll(12));
// console.log(sumAll(1, 2));
// console.log(sumAll(1, 2, 3, 4, 5));