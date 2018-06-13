# These parameters work well for large scale images
    # params = convolution_parameters(road_width=35, road_length=200,
    #                                 peak_min_distance=100,
    #                                 kernel_type=ktype.GAUSSIAN)

    # These parameters work well for smaller images
    # params = convolution_parameters(road_width=30, road_length=70,
    #                                 peak_min_distance=150,
    #                                 kernel_type=ktype.NEGATIVE)


    # Params for section 1
    # feature = RoadIntersectionDensity(image_path, road_width=20,
    #                                   road_length=70,
    #                                   peak_min_distance=100,
    #                                   kernel_type=ktype.GAUSSIAN,
    #                                   scale=100,
    #                                   block_size=20).visualize()

    # Params for section 3
    # feature = RoadIntersectionDensity(image_path, road_width=30,
    #                                   road_length=70,
    #                                   peak_min_distance=150,
    #                                   kernel_type=ktype.GAUSSIAN,
    #                                   scale=150,
    #                                   block_size=20).visualize()
