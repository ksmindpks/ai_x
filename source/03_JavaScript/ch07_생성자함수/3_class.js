class Student{
    constructor(name, kor, mat, eng, sci){
        this.name = name;
        this.kor = kor;
        this.eng = eng;
        this.mat = mat;
        this.sci = sci;
    }
    getSum(){
        return this.kor + this.mat + this.eng + this.sci;
    }
    getAvg(){
        return this.getSum() / 4;
    }
    toString(){
        return 'name:'+this.name+
                ' kor:'+this.kor+
                ' mat:'+this.mat+
                ' eng:'+this.eng+
                ' sci:'+this.sci+
                ' 합:'+this.getSum()+
                ' 평균:'+this.getAvg();
    } // class
}
var hong = new Student("홍", 99, 98, 97, 96);

console.log(hong.kor);
console.log(hong.toString);
console.log(`${hong}`);
console.log(hong); // 자동 템플릿 리터럴이 호출