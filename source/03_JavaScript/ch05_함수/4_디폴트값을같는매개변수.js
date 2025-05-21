console.log(pow(7,3));
console.log(pow(9));
console.log(pow());

function pow(x=1, y=2){
    // x의 y승을 return
    var rslt = 1;
    for(let cnt=1; cnt<=y; cnt++){
        rslt *= x; // rslt=rslt*x;
    }
    return rslt;
}

console.log(pow(7,3));