import 'package:flutter/material.dart';

class Login extends StatefulWidget {
  const Login({super.key});

  @override
  State<Login> createState() => _LoginState();
}

class _LoginState extends State<Login> {

  TextEditingController email = TextEditingController();
  TextEditingController password = TextEditingController();  

    signIn() async {
        await FirebaseAuth.instance.signInWithEmailAndPassword(
            email: email.text,
            password: password.text,
  );
}


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Login"),),
      body:Padding(
        padding: const EdgeInsets.all(20.0)
        children: [
            TextField(
                controller: email,
                decoration: InputDecoration(hintText: 'Enter email',),
            ),
            TextField(
                controller: password,
                decoration: InputDecoration(hintText: 'Enter password',),
            ),

            ElevatedButton(onPressed: (()=>signIn()), child: Text("Login"))            
        ],
       )

    );
  }
}