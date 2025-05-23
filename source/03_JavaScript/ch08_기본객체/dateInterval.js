Date.prototype.getIntervalDay = function(otherday){
    // this와 otherday 사이의 날짜 수를 return
    // now.getIntervalDay(openday): this가 now, otherday가 openday
    // openday.getIntervalDay(now) : this가 openday, otherday가 now
    // if(this > otherday){
    //     var intervalMiliSec = this.getTime() - otherday.getTime();
    // }else{
    //     var intervalMiliSec = otherday.getTime() - this.getTime();
    // }
    // var intervalMiliSec = (this>other)?
    //                         (this.getTime() - otherday.getTime()):
    //                         (otherday.getTime() - this.getTime());
    var intervalMiliSec = Math.abs(this.getTime() - otherday.getTime())
    let day = intervalMiliSec/(1000*60*60*24);
    return Math.trunc(day); // Math.trunc: 내림,  Math.round: 반올림, Math.ceil: 올림림
};

// var now = new Date();
// var openday = new Date(2025, 3, 7, 9, 30, 0);
// console.log(now.getIntervalDay(openday), '일');
// console.log(openday.getIntervalDay(now), '일');