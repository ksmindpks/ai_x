// false로 간주되는 값 : undefined, 0, NaN, '', Null
// Boolean(값) : boolena으로 형변환
// Number(값) : number로 형변환
// String(값) : string으로 형변환
var i;
console.log(Boolean(i));
console.log(Boolean(0));
console.log(Boolean(NaN));
console.log(Boolean(Number('a')));
console.log(Boolean(''));
console.log(Boolean(' ')); // true
console.log(Boolean(null));