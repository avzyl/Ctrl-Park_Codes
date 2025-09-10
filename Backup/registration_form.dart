import 'package:ctrlpark_app/features/authentication/screens/login/widgets/login_form.dart';
import 'package:ctrlpark_app/features/authentication/screens/registration/widgets/student_info_form.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:ctrlpark_app/utils/constants/sizes.dart';
import 'package:ctrlpark_app/utils/constants/text_strings.dart';
import 'package:ctrlpark_app/features/authentication/screens/registration/widgets/personal_info_form.dart';
import '../../../../common/widgets/login_signup/form_divider.dart';
import '../../../../common/widgets/success_screen/success_screen.dart';

class SignupScreen extends StatefulWidget {
  const SignupScreen({super.key});

  @override
  State<SignupScreen> createState() => _SignupScreenState();
}

class _SignupScreenState extends State<SignupScreen> {
  int _currentPage = 0;
  final _formKey = GlobalKey<FormState>();
  bool _agreedToTerms = false;

  /// Subtitles for each page
  final List<String> _subtitles = [
    'Driver Information',
    'Student Information',
    'Vehicle Information',
  ];

  /// Form Pages
  List<Widget> get _formPages => [
    /// Page 1: Personal Info
    const PersonalInfoForm(),

    /// Page 2: Student Info
    const StudentInfoForm(),

    /// Page 3: Car Info
    Column(
      children: [
        // make & model
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.makeAndModel,
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        // color
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.carColor,
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        // license plate no
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.licensePlateNo,
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        // type
        TextFormField(
          decoration: const InputDecoration(
            labelText: CPTexts.carType,
          ),
        ),
        const SizedBox(height: CPSizes.spaceBtwInputFields),

        Row(
          children: [
            Checkbox(
              value: _agreedToTerms,
              onChanged: (value) {
                setState(() {
                  _agreedToTerms = value ?? false;
                });
              },
            ),
            Expanded(
              child: Text(
                CPTexts.agreement,
                style: Theme.of(context).textTheme.bodySmall,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ],
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          CPTexts.signInButton,
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        iconTheme: IconThemeData(
          color: Theme.of(context).brightness == Brightness.dark
              ? Colors.blueAccent
              : Colors.black,
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(CPSizes.defaultSpace),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Title
              Text(
                CPTexts.registerTitle,
                style: Theme.of(context).textTheme.headlineMedium,
              ),
              Text(
                _subtitles[_currentPage],
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: CPSizes.spaceBtwSections),

              // Form
              Form(
                key: _formKey,
                child: _formPages[_currentPage],
              ),
              const SizedBox(height: CPSizes.spaceBtwSections),

              // Navigation Buttons
              Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Submit or Continue Button
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: () {
                        if (_formKey.currentState!.validate()) {
                          if (_currentPage < _formPages.length - 1) {
                            setState(() => _currentPage++);
                          } else {
                            if (!_agreedToTerms) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content: Text(
                                      'Please agree to the Terms and Conditions'),
                                ),
                              );
                              return;
                            }
                            Get.to(() => const SuccessScreen());
                          }
                        }
                      },
                      child: Text(
                        _currentPage == _formPages.length - 1
                            ? 'Submit'
                            : 'Continue',
                      ),
                    ),
                  ),
                  const SizedBox(height: CPSizes.spaceBtwInputFields),

                  // Back Button
                  if (_currentPage > 0)
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        onPressed: () => setState(() => _currentPage--),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.grey.shade300,
                          foregroundColor: Colors.black,
                        ),
                        child: const Text('Back'),
                      ),
                    ),

                  // Divider and Social Buttons (only on first page)
                  if (_currentPage == 0) ...[
                    const SizedBox(height: CPSizes.spaceBtwSections),
                    CPFormDivider(
                      dividerText: CPTexts.orSignUpWith.capitalize!,
                    ),
                    const SizedBox(height: CPSizes.spaceBtwSections),
                    const CPSocialButtons(),
                  ],
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
