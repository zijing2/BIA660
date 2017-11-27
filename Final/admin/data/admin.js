// const mongoCollections = require("../config/mongoCollections");
// const users = mongoCollections.users;
const uuid = require('node-uuid');
const bcrypt = require("bcrypt-nodejs");

var admin = [
    {
        "adminname":"admin",
        "password":bcrypt.hashSync("admin")
    }
];

let exportedMethods = {
    
    checkAdminLogin(adminname,password){
        return new Promise((resolve, reject)=>{
            for (var i = 0; i < admin.length; i++) {
                var element = admin[i];
                if(element.adminname==adminname&&bcrypt.compareSync(password, element.password)){
                    resolve(element);
                }
            }
                reject("invalid username or password");
        });
        
    }
    
    

}

module.exports = exportedMethods;   