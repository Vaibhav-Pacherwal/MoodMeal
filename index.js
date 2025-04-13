const express = require("express");
const ejs = require("ejs");
const bodyParser = require("body-parser");
const { exec } = require("child_process");
const path = require("path");
const { stdout, stderr } = require("process");
const { error } = require("console");

const app = express();
let port = 8080;

app.listen(port, (req, res)=>{
    console.log(`server is running on https://localhost:${port}`);
})

app.set("view engine", "ejs");
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static("public"));

app.get('/', (req, res)=>{
    res.render("home");
});

app.post('/predict', (req, res)=>{
    const userInput = req.body.userInput;
    const command = `python predict.py ${userInput}`;

    exec(command, (error, stdout, stderr)=>{
        if(error) {
            console.error(error.message);
            res.send("Error occured during prediction!!")
        }

        if(stderr) {
            console.log("stderr", stderr);
        }

        const output = stdout.trim();
        const [mood, meal, recipe, ytLink] = output.split("::");
        res.render("result", {
            mood,
            meal,
            userInput,
            recipe,
            ytLink
        });
    })
})


