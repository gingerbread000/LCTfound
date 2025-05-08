                        '''
                        if sigma_t >= at_next*sigma_y:
                            lambda_t = 1.
                            gamma_t = 0 # (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
                        else:
                            lambda_t = (sigma_t)/(at_next*sigma_y)
                            gamma_t = 0.
                        '''
                        # print('T : ',t)
                        '''
                        lambda_t = self.args.lambda_t/1000*t
                        if lambda_t==0:
                            lambda_t = 0.000001*torch.ones(lambda_t.shape).cuda()
                        '''

                        '''
                        def test_sigmoid(x, b):
                            import math
                            a=-math.log(9)
                            c=0.1
                            y=(1/(1+math.exp(-(b*x+a)))-c)/(1-c)
                            print('Y ---> ',y)
                            # torch.from_numpy(y).cuda()
                            y_output = torch.ones(1)*y
                            return y_output.cuda()s
                        lambda_t = test_sigmoid(t, self.args.lambda_t)
                        '''


                        '''
                        if t>200:
                            lambda_t = self.args.lambda_t/1000*t
                        else:
                            lambda_t = 0.00001
                        '''

                                                    # print('lambda_t : ',lambda_t.cpu().detach().numpy())

                            # print(f"\nj: {j}, lambda_t: {lambda_t}, at_next: {at_next.cpu().numpy()}, sigma_t: {sigma_t.cpu().numpy()}")
                            # if len(lambda_t.shape)==4:
                            #     lambda_ts.append(lambda_t.cpu().numpy()[0, 0, 0, 0] if not isinstance(lambda_t, float) else lambda_t)
